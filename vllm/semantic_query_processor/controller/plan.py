from ..sem_ops import BaseOp, OpKind, base, ops

from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.data import data
from vllm.semantic_query_processor.budget import KVMemoryManager
from vllm.semantic_query_processor.execution.pipeline_execution import PlanExecutor
from .pipeline import SemanticPipeline
from .pipeline import pipeline_builder, is_pipeline_builder
from collections import defaultdict

class SemanticPlan:

    def __init__(self, executor):
        self.executor = executor
        self.plan_executor = PlanExecutor()


    def build(self, ctxs, operators):

        plan = []

        current_segment = []          # [pipeline_stage, cp, pipeline_stage, ...]
        pipeline_ops_chain = []       # tuple-independent ops to fuse
        next_stage_id = 1             # global unique stage id

        def emit_pipeline():
            nonlocal pipeline_ops_chain, current_segment, next_stage_id

            if not pipeline_ops_chain:
                return

            pipeline_stage = pipeline_builder(
                tuple(pipeline_ops_chain),
                next_stage_id
            )

            current_segment.append(pipeline_stage)

            next_stage_id += 1
            pipeline_ops_chain = []

        def flush_segment():
            nonlocal current_segment

            if not current_segment:
                return

            # collect stage_ids in this segment
            stage_ids = [
                item.stage_id
                for item in current_segment
                if hasattr(item, "stage_id")
            ]

            kv = KVMemoryManager.get_instance()
            num_stages = len(stage_ids)

            if num_stages == 1:
                kv.register_stage(stage_ids[0], 1.0)

            elif num_stages > 1:
                primary_ratio = 0.8
                remaining = 1.0 - primary_ratio

                kv.register_stage(stage_ids[0], primary_ratio)

                share = remaining / (num_stages - 1)
                for sid in stage_ids[1:]:
                    kv.register_stage(sid, share)

            plan.append(current_segment)
            current_segment = []

        for op in operators:

            if op.kind == OpKind.TUPLE_INDEPENDENT:
                pipeline_ops_chain.append(op)
                continue

            # boundary encountered
            emit_pipeline()

            if op.kind == OpKind.JOIN:  # CP
                current_segment.append(op)

            elif op.kind == OpKind.BLOCKING:
                flush_segment()
                plan.append(op)

            else:
                raise ValueError(f"Unknown op kind: {op.kind}")

        # tail flush
        emit_pipeline()
        flush_segment()

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
            ops.SemAgg("Find the common opinion"),
            ops.SemFilter("Is the research paper related to the given category?"),
        )

        plan = self.build(ctxs, jf_operators)
        self.print_plan(plan)
                
        out = await self.plan_executor.execute(ctxs, plan)
        return out 

