from ..sem_ops import BaseOp, OpKind, base, ops

from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.data import data
from vllm.semantic_query_processor.budget import KVMemoryManager
from vllm.semantic_query_processor.execution.pipeline_execution import AsyncPipelineExecutor
from .pipeline import SemanticPipeline
from .pipeline import pipeline_builder, is_pipeline_builder
from collections import defaultdict

class SemanticPlan:

    def __init__(self, executor):
        self.executor = executor

    def plan_resource(self, ctxs, plan): 
        manager = KVMemoryManager.get_instance()
        for pipeline in plan:
            if isinstance(pipeline, BaseOp):
                continue
        # check left/right table on a pipeline and allocate resources accordingly
        manager.register_stage(1, 0.8)
        manager.register_stage(2, 0.2)
            

            
        return "resource allocation done print some stuff"


    def build(self, ctxs, operators):

        plan = []
        pipeline_ops = []

        stage_id = 1

        for idx, op in enumerate(operators):


            if op.kind == OpKind.TUPLE_INDEPENDENT:
                pipeline_ops.append(op)
                continue

            # flush chain before blocking op
            if pipeline_ops: 
                plan.append(
                    pipeline_builder(tuple(pipeline_ops), stage_id)
                )
                pipeline_ops = []
  
            plan.append(op)

            if op.kind == OpKind.BLOCKING:
                stage_id = 1
            elif isinstance(op, ops.CartesianProduct):
                stage_id += 1


        # flush tail
        if pipeline_ops:
            plan.append(
                pipeline_builder(tuple(pipeline_ops), stage_id)
            )

        resource_allocation = self.plan_resource(ctxs, plan)
        # register quotas here
 
        return plan



    def print_plan(self, plan):
        for stage in plan:
            if is_pipeline_builder(stage):
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
            ops.SemFilter("Is the research paper related to the given category?"),
        )

        plan = self.build(ctxs, jf_operators)
        self.print_plan(plan)
                
        pipe = []
        # ctxs = ctxs[:1]
        for stage in plan:
            
            if not (isinstance(stage, ops.BaseOp) and stage.kind == OpKind.BLOCKING):
                pipe.append(stage)
            else:
                ctxs = await self._execute_plan(ctxs, pipe)
                ctxs = await stage(ctxs)
                pipe = []

        if pipe:
            ctxs = await self._execute_plan(ctxs, pipe)


        print('len(ctxs)', len(ctxs))
        print(f"pipeline finished {len(ctxs)} results")

        return ctxs


    async def _execute_plan(self, ctxs, pipeline):
        return await AsyncPipelineExecutor().execute_tasks(ctxs, pipeline)