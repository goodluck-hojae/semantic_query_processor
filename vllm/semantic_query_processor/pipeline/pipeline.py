from ..sem_ops import base, ops

from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.data import data
from vllm.semantic_query_processor.pipeline import SemanticChain
import asyncio


class SemanticPipeline:

    def __init__(self, kv_estimator):
        self.kv_estimator = kv_estimator

    def build(self, operators):
        """
        Returns:
            [
                (chain_builder, blocking_op),
                ...
            ]
        """

        def build_chain(chain_ops_snapshot):
            def _builder(ctx):
                return SemanticChain(
                    ctx,
                    *chain_ops_snapshot,
                    bytes_per_token=self.kv_estimator.bytes_per_token,
                )
            return _builder

        pipeline = []
        chain_ops = []

        for op in operators:
            if op.kind == base.OpKind.TUPLE_INDEPENDENT:
                chain_ops.append(op)
            else:
                pipeline.append(
                    (build_chain(tuple(chain_ops)), op)
                )
                chain_ops = []

        if chain_ops:
            pipeline.append(
                (build_chain(tuple(chain_ops)), None)
            )

        return pipeline


    async def execute(self, raw_request, query: Query):
        ctxs = list(data._data_source(raw_request, query, self.kv_estimator))

        operators = (
            ops.SemFilter("Is the candidate capable of GPU programming?", pin=True),
            ops.SemMap("Summarize the following resume.", is_last=True),
            ops.SemMap("Test", is_last=True),
            ops.SemTopK(ctxs),
            ops.SemFilter("Is the candidate capable of GPU programming?", pin=True),
            ops.SemMap("Summarize the following resume.", is_last=True),
        )

        pipeline = self.build(operators)

        for chain_builder, blocking_op in pipeline:
            if chain_builder is not None:
                ctxs = await self._execute_chain(ctxs, chain_builder)

            if blocking_op is not None:
                ctxs = await blocking_op(ctxs)

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