import asyncio
from vllm.semantic_query_processor.budget import KVMemoryManager

from ..sem_ops import ops


class AsyncPipelineExecutor:
    def __init__(self):
        self.manager = KVMemoryManager.get_instance()

    async def execute_tasks(self, ctxs, pipelines):
        """
        ctxs: initial contexts
        pipelines: [pipeline1, cp, pipeline2, ...]
        """
 
        async def release_allocs(allocs): 
            for stage_id, budget in allocs:
                await self.manager.release(stage_id, budget)

        async def run_pipeline_stage(ctx, stage, owned_allocs):
            
            pipeline = stage(ctx)
            await self.manager.acquire(pipeline.stage_id, pipeline.budget)
            owned_allocs.append((pipeline.stage_id, pipeline.budget))
            await pipeline()
            return pipeline.ctx

        async def run_ctx(ctx, remaining, pinned_allocs):
            owned_allocs = []
            try:
                i = 0
                cur_ctx = ctx

                while True:
                    if i >= len(remaining):
                        return cur_ctx

                    stage = remaining[i]

                    # normal pipeline stage
                    if not isinstance(stage, ops.CartesianProduct):
                        cur_ctx = await run_pipeline_stage(cur_ctx, stage, owned_allocs)
                        i += 1
                        continue

                    # CartesianProduct: parent spawns children for the rest
                    new_ctxs = stage(cur_ctx)
                    rest = remaining[i + 1 :]

                    if not new_ctxs:
                        return None  
                    
                    child_pinned = pinned_allocs + owned_allocs

                    # Spawn children and wait for all, then parent can unpin its owned_allocs
                    child_tasks = [asyncio.create_task(run_ctx(c, rest, child_pinned)) for c in new_ctxs]
                    child_results = await asyncio.gather(*child_tasks, return_exceptions=False)

                    # Flatten results; filter out None
                    out = [r for r in child_results if r is not None]

                    if cur_ctx.state.pin_req_id is not None:
                        await cur_ctx.state.executor.unpin(cur_ctx.state.raw_request, cur_ctx.state.pin_req_id)
                        cur_ctx.state.pin_req_id = None
                        
                    return out

            finally:
                await release_allocs(owned_allocs)

        root_tasks = [asyncio.create_task(run_ctx(ctx, pipelines, pinned_allocs=[])) for ctx in ctxs]
        root_results = await asyncio.gather(*root_tasks, return_exceptions=False)

        out = []
        for r in root_results:
            if r is None:
                continue
            if isinstance(r, list):
                out.extend(r)
            else:
                out.append(r)

        return out