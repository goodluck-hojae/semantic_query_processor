from ..sem_ops import BaseOp, base, ops

from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.data import data
from vllm.semantic_query_processor.budget import KVMemoryManager
from .pipeline import SemanticPipeline
from collections import defaultdict

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


    def print_plan(self, plan):
        for stage in plan:
            if callable(stage) and not isinstance(stage, BaseOp):
                print(str(stage.ops))
                continue
            print(str(stage))

    async def execute(self, raw_request, query: Query):
        ctxs = list(data._data_source(raw_request, query, self.executor))

        # Filter-Filter
        ff_operators = (
            ops.SemFilter("The review contains substantive content, meaning it is not short (less than three sentences) or vague and expresses a concrete opinion about the movie", pin=True),
            ops.SemFilter("The review criticizes the movie’s plot, storytelling, or narrative structure"),
        )
        # Filter-Filter-Map
        ffm_operators = (
            ops.SemFilter("The review contains substantive content, meaning it is not short (less than three sentences) or vague and expresses a concrete opinion about the movie", pin=True),
            ops.SemFilter("The review criticizes the movie’s plot, storytelling, or narrative structure"),
            ops.SemMap("Summarize the review"),
        )
 
        # Map-Filter-Filter
        mff_operators = (
            ops.SemMap("Summarize the review", pin=True),
            ops.SemFilter("The review contains substantive content, meaningful or vague and expresses a concrete opinion about the movie"),
            ops.SemFilter("The review criticizes the movie’s plot, storytelling, or narrative structure", unpin=True),
        )

        # Filter - GroupBy - Aggregation
        groups = ["Positive", "Negative"]
        fga_operators = (
            ops.SemFilter("The review contains substantive content, meaning it is not short (less than three sentences) or vague and expresses a concrete opinion about the movie", pin=True),              
            ops.SemGroupBy(groups, unpin=True),   
            ops.SemAgg("Find the common opinion"),
        )

        # Join-Filter
        research_categories = data.research_category_data()
        jf_operators = (
            ops.CartesianProduct(right_table=research_categories),
            ops.SemFilter("Is the research paper related to the given category?", pin=True),
            ops.SemMap("Summarize the research abstract and explain how it is related to the category"),
        )

        plan = self.build(jf_operators)
        self.print_plan(plan)
                
        
        for stage in plan:
            # chain
            if callable(stage) and not isinstance(stage, BaseOp):
                print(f"{str(stage.ops)} processing")
                ctxs = await self._execute_plan(ctxs, stage)
                continue

            # blocking op
            print(f"{str(stage)} processing")
            ctxs = await stage(ctxs)

            print('len(ctxs)', len(ctxs))
        print(f"pipeline finished {len(ctxs)} results")

        return ctxs


    async def _execute_plan(self, ctxs, chain_builder, concurrency=100):
        return await KVMemoryManager.get_instance().execute_tasks(
            seeds=ctxs,
            task_builder=chain_builder,
            concurrency=concurrency,
        )