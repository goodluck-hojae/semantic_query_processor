from ..sem_ops import BaseOp, OpKind, base, ops

from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.data import data
from vllm.semantic_query_processor.budget import KVMemoryManager
from .pipeline import SemanticPipeline
from .pipeline import pipeline_builder
from collections import defaultdict

class SemanticPlan:

    def __init__(self, executor):
        self.executor = executor

    def build(self, operators):

        pipeline = []
        pipeline_ops = []

        # find CartesianProduct index in operator list
        join_index = None
        for i, op in enumerate(operators):
            if isinstance(op, ops.CartesianProduct):
                join_index = i
                break

        for idx, op in enumerate(operators):

            if op.kind == OpKind.TUPLE_INDEPENDENT:
                pipeline_ops.append(op)
                continue

            # flush chain before blocking op
            if pipeline_ops:
                if join_index is not None and idx <= join_index:
                    stage_id = "pipeline_1"
                else:
                    stage_id = "pipeline_2"

                pipeline.append(
                    pipeline_builder(tuple(pipeline_ops), stage_id)
                )
                pipeline_ops = []

            pipeline.append(op)

        # flush tail
        if pipeline_ops:
            if join_index is not None and len(operators) > join_index:
                stage_id = "pipeline_2"
            else:
                stage_id = "pipeline_1"

            pipeline.append(
                pipeline_builder(tuple(pipeline_ops), stage_id)
            )

        # register quotas here
        manager = KVMemoryManager.get_instance()
        if join_index is not None:
            manager.register_stage("pipeline_1", 0.05)
            manager.register_stage("pipeline_2", 0.95)
        else:
            manager.register_stage("pipeline_2", 1.0)

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
            ops.SemMap("Summarize the research abstract and explain how it is related to the category", pin=True),
            ops.CartesianProduct(right_table=research_categories),
            ops.SemMap("Is the research paper related to the given category?"),
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