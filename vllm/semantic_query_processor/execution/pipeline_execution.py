import asyncio
from vllm.semantic_query_processor.budget import KVMemoryManager
from vllm.semantic_query_processor.sem_ops import OpKind, ops
from collections import defaultdict
class PlanExecutor:

    def __init__(self):
        self.pipeline_executor = AsyncPipelineExecutor()
        self.blocking_executor = BlockingExecutor()


    async def execute(self, ctxs, plan):

        stage_stat_list = []
        for item in plan:

            # BLOCKING
            if isinstance(item, ops.BaseOp) and item.kind == OpKind.BLOCKING:
                ctxs = await item(ctxs)
                
            # PIPELINE
            else:
                ctxs, stage_stat = await self.pipeline_executor.execute_tasks(
                    ctxs,
                    item
                )
                stage_stat_list.append(stage_stat)

        return ctxs, stage_stat_list


class AsyncPipelineExecutor:
    def __init__(self):
        self.manager = KVMemoryManager.get_instance()

    async def execute_tasks(self, ctxs, pipelines):
        """
        ctxs: initial contexts
        pipelines: [pipeline1, cp, pipeline2, ...]
        """
        stage_stat = {}
 
        async def release_allocs(allocs): 
            for stage_id, budget in allocs:
                await self.manager.release_stage(stage_id, budget)

        async def run_pipeline_stage(ctx, stage, local_allocs):
            
            pipeline = stage(ctx)
            await self.manager.allocate_stage(pipeline.stage_id, pipeline.budget)
            local_allocs.append((pipeline.stage_id, pipeline.budget))
            result = await pipeline()
            return result

        async def run_ctx(ctx, remaining, inherited_allocs):
            local_allocs = []
            try:
                i = 0
                cur_ctx = ctx

                while True:
                    if i >= len(remaining):
                        return cur_ctx

                    stage = remaining[i]

                    # normal pipeline stage
                    if not isinstance(stage, ops.CartesianProduct):
                        if stage.stage_id in stage_stat:
                            stage_stat[stage.stage_id]["input"] += 1
                        else:
                            stage_stat[stage.stage_id] = {"input": 1, "output": 0}

                        cur_ctx = await run_pipeline_stage(cur_ctx, stage, local_allocs)
                        if not cur_ctx:
                            return None  
                        stage_stat[stage.stage_id]["output"] += 1
                        i += 1
                        continue

                    # CartesianProduct: parent spawns children for the rest
                    new_ctxs = stage(cur_ctx)
                    rest = remaining[i + 1 :]

                    if not new_ctxs:
                        return None  
                    
                    child_inherited_allocs = inherited_allocs + local_allocs

                    # Spawn children and wait for all, then parent can unpin its local_allocs
                    child_tasks = [asyncio.create_task(run_ctx(c, rest, child_inherited_allocs)) for c in new_ctxs]
                    child_results = await asyncio.gather(*child_tasks, return_exceptions=False)

                    # Flatten results; filter out None
                    out = [r for r in child_results if r is not None]

                    if cur_ctx.state.pin_req_id is not None:
                        await cur_ctx.state.executor.unpin(cur_ctx.state.raw_request, cur_ctx.state.pin_req_id)
                        cur_ctx.state.pin_req_id = None
                        
                    return out

            finally:
                await release_allocs(local_allocs)

        root_tasks = [asyncio.create_task(run_ctx(ctx, pipelines, inherited_allocs=[])) for ctx in ctxs]
        root_results = await asyncio.gather(*root_tasks, return_exceptions=False)
    
        out = []
        for r in root_results:
            child_results = [r]
            while child_results:
                item = child_results.pop()
                if item is None:
                    continue
                if isinstance(item, list):
                    child_results.extend(item)
                else:
                    out.append(item)

        return out, stage_stat
    


class BlockingExecutor:

    @staticmethod
    async def execute_tasks(
        seeds,
        task_builder,
        concurrency: int = 100,
    ):
        manager = KVMemoryManager.get_instance()

        queue = asyncio.Queue(maxsize=concurrency)
        capacity_cond = asyncio.Condition()
        results = []

        async def worker():
            while True:
                task = await queue.get()
                try:
                    out = await task()
                    results.append(out)
                finally:
                    async with capacity_cond:
                        await manager.release(task.budget)
                        capacity_cond.notify_all()
                    queue.task_done()

        workers = [
            asyncio.create_task(worker())
            for _ in range(concurrency)
        ]

        for seed in seeds:
            task = task_builder(seed)

            async with capacity_cond:
                while not await manager.can_admit(task.budget):
                    await capacity_cond.wait()
                await manager.allocate(task.budget)

            await queue.put(task)

        await queue.join()

        for w in workers:
            w.cancel()

        return results
