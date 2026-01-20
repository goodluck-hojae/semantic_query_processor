from ..sem_ops import BaseOp, base, ops

from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.data import data
from vllm.semantic_query_processor.pipeline import SemanticChain
import asyncio


class SemanticPipeline:

    def __init__(self, kv_estimator):
        self.kv_estimator = kv_estimator

    def build(self, operators):

        def chain_builder(chain_ops_snapshot):
            def _chain(ctx):
                return SemanticChain(
                    ctx,
                    *chain_ops_snapshot,
                    bytes_per_token=self.kv_estimator.bytes_per_token,
                )
            return _chain

        pipeline = []
        chain_ops = []

        for op in operators:
            if op.kind == base.OpKind.TUPLE_INDEPENDENT:
                chain_ops.append(op)
                continue

            # flush any pending chain
            if chain_ops:
                pipeline.append(chain_builder(tuple(chain_ops)))
                chain_ops = []

            # emit blocking op directly (JOIN, TopK, etc.)
            pipeline.append(op)

        # flush tail chain
        if chain_ops:
            pipeline.append(chain_builder(tuple(chain_ops)))

        return pipeline


    async def execute(self, raw_request, query: Query):
        ctxs = list(data._data_source(raw_request, query, self.kv_estimator))

        operators = (
            ops.SemFilter("Is the candidate capable of GPU programming?", pin=True),
            ops.SemFilter("Is the candidate capable of GPU programming?"),
            ops.SemFilter("Is the candidate capable of GPU programming?", unpin=True),
            ops.SemJoin(ctxs),
            ops.SemFilter("Is the candidate capable of GPU programming?", pin=False),
        )

        pipeline = self.build(operators)

        for stage in pipeline:
            # chain
            print(f"{str(stage)} processing")
            if callable(stage) and not isinstance(stage, BaseOp):
                ctxs = await self._execute_chain(ctxs, stage)
                continue

            # blocking op
            ctxs = await stage(ctxs)
            print('len(ctxs)', len(ctxs))
            ctxs = ctxs[:100]
        print(f"pipeline finished")
        return ctxs


    async def _execute_chain(self, ctxs, chain_builder, concurrency=100):
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
                        self.kv_estimator.release(chain.budget)
                        capacity_cond.notify_all()
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]

        for ctx in ctxs:
            chain = chain_builder(ctx)

            async with capacity_cond:
                while not self.kv_estimator.can_admit(chain.budget):
                    await capacity_cond.wait()
                self.kv_estimator.allocate(chain.budget)

            await queue.put(chain)

        await queue.join()

        for w in workers:
            w.cancel()

        return results