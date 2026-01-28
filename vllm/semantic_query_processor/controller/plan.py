from ..sem_ops import BaseOp, base, ops

from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.data import data
from vllm.semantic_query_processor.budget import KVMemoryManager
from .pipeline import SemanticPipeline


class SemanticPlan:

    def __init__(self, executor):
        self.executor = executor

    def build(self, operators):

        def pipeline_builder(ops):
            def _pipeline(ctx):
                return SemanticPipeline(
                    ctx,
                    *ops,
                    bytes_per_token=KVMemoryManager.get_instance().bytes_per_token,
                )
            
            _pipeline.ops = ops
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
        ctxs = list(data._data_source(raw_request, query, self.executor))
        operators = (
            # ops.SemFilter("Does this candidate have Computer Science degree?", pin=True),
            # ops.SemFilter("Is the candidate capable of GPU programming?", unpin=True),
            ops.SemMap("Summarize the resume", use_output_as_prompt=False),
            ops.SemMap("Summarize the resume", expand=True, max_len=2096, use_output_as_prompt=True, pin=True),
            ops.SemMap("Summarize the resume", expand=False, use_output_as_prompt=True, pin=False),
            
        )

        pipeline = self.build(operators)

        for stage in pipeline:
            # chain
            print(f"{str(stage)} processing")
            if callable(stage) and not isinstance(stage, BaseOp):
                ctxs = await self._execute_plan(ctxs, stage)
                continue

            # blocking op
            ctxs = await stage(ctxs)
            print('len(ctxs)', len(ctxs))
            ctxs = ctxs[:50]
        print(f"pipeline finished")
        return ctxs


    async def _execute_plan(self, ctxs, chain_builder, concurrency=100):
        return await KVMemoryManager.get_instance().execute_tasks(
            seeds=ctxs,
            task_builder=chain_builder,
            concurrency=concurrency,
        )