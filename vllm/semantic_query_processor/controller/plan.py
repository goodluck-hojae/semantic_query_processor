from ..sem_ops import BaseOp, base, ops

from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.data import data
from vllm.semantic_query_processor.budget import KVBudget
from .pipeline import SemanticPipeline
import asyncio


class SemanticPlan:

    def __init__(self):
        pass

    def build(self, operators):

        def pipeline_builder(chain_ops_snapshot):
            def _pipeline(ctx):
                return SemanticPipeline(
                    ctx,
                    *chain_ops_snapshot,
                    bytes_per_token=KVBudget.get_instance().bytes_per_token,
                )
            return _pipeline

        pipeline = []
        pipeline_ops = []

        for op in operators:
            if op.kind == base.OpKind.TUPLE_INDEPENDENT:
                pipeline_ops.append(op)
                continue

            # flush any pending chain
            if pipeline_ops:
                pipeline.append(pipeline_builder(tuple(pipeline_ops)))
                pipeline_ops = []

            # emit blocking op directly (JOIN, TopK, etc.)
            pipeline.append(op)

        # flush tail chain
        if pipeline_ops:
            pipeline.append(pipeline_builder(tuple(pipeline_ops)))

        return pipeline


    async def execute(self, raw_request, query: Query):
        ctxs = list(data._data_source(raw_request, query))

        operators = (
            ops.SemAgg("Summarize the trend of resumes", 10000),
        )

        pipeline = self.build(operators)

        for stage in pipeline:
            # chain
            print(f"{str(stage)} processing")
            if callable(stage) and not isinstance(stage, BaseOp):
                ctxs = await self._execute_pipeline(ctxs, stage)
                continue

            # blocking op
            ctxs = await stage(ctxs)
            print('len(ctxs)', len(ctxs))
            ctxs = ctxs[:100]
        print(f"pipeline finished")
        return ctxs


    async def _execute_pipeline(self, ctxs, chain_builder, concurrency=100):
        queue = asyncio.Queue(maxsize=concurrency)
        capacity_cond = asyncio.Condition()
        results = []

        async def worker():
            while True:
                chain = await queue.get()
                try:
                    await chain()
                    results.append(chain.ctx)
                finally:
                    async with capacity_cond:
                        KVBudget.get_instance().release(chain.budget)
                        capacity_cond.notify_all()
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]

        for ctx in ctxs:
            chain = chain_builder(ctx)

            async with capacity_cond:
                while not KVBudget.get_instance().can_admit(chain.budget):
                    await capacity_cond.wait()
                KVBudget.get_instance().allocate(chain.budget)

            await queue.put(chain)

        await queue.join()

        for w in workers:
            w.cancel()

        return results