from vllm.semantic_query_processor.sem_ops import OpKind, OpName, ops
from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.data import data
from vllm.semantic_query_processor.budget import KVMemoryManager
from vllm.semantic_query_processor.execution.pipeline_execution import PlanExecutor
from vllm.semantic_query_processor.controller.map_estimator import MapRatioEstimator
from .pipeline import pipeline_builder, is_pipeline_builder
import math

class SemanticPlan:

    def __init__(self, executor):
        self.executor = executor
        self.plan_executor = PlanExecutor()
        


    def build(self, ctxs, operators):

        plan = []

        current_segment = []          # [pipeline_stage, cp, pipeline_stage, ...]
        pipeline_ops_chain = []       # tuple-independent ops to fuse
        next_stage_id = 1

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

            
            # initially allocate resource equally
            # And warm up for allocation
            elif num_stages > 1:
                for sid in stage_ids[0:]:
                    kv.register_stage(sid, 1 / num_stages)

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
        for i, item in enumerate(plan, start=1):
            # PIPELINE: [pipe, cp, pipe, ...]
            if isinstance(item, list):
                print(f"[{i}] PIPELINE")
                for j, stage in enumerate(item, start=1):
                    if is_pipeline_builder(stage):
                        op_names = [op.__class__.__name__ for op in stage.ops]
                        print(
                            f"  ({j}) PIPELINE stage_id={stage.stage_id} "
                            f"ops={op_names}"
                        )
                    else:
                        print(f"  ({j}) {stage.__class__.__name__}")
            else:
                # Blocking op
                print(f"[{i}] BLOCKING {item.__class__.__name__}")


    def parse_ops(self, logical_ops):

        def apply_pin_unpin(ops_list):

            # Reset flags
            for op in ops_list:
                op.pin = False
                op.unpin = False

            chain = []

            def finalize_chain(next_kind):
                nonlocal chain
                if not chain:
                    return

                first = chain[0]

                # Chain feeds CP
                if next_kind == OpKind.JOIN:
                    first.pin = True

                # End of query
                elif next_kind is None:
                    if len(chain) > 1:
                        last = chain[-1]
                        first.pin = True
                        last.unpin = True

                chain = []

            n = len(ops_list)

            for i, op in enumerate(ops_list):
                next_op = ops_list[i + 1] if i + 1 < n else None
                next_kind = next_op.kind if next_op else None

                if op.kind == OpKind.TUPLE_INDEPENDENT:
                    chain.append(op)

                    # boundary detected
                    if next_op is None or next_kind != OpKind.TUPLE_INDEPENDENT:
                        finalize_chain(next_kind)

                else:
                    chain = []

            for op in ops_list:
                print(f"op: {op}, pin: {op.pin} unpin: {op.unpin}")
                assert not (op.pin and op.unpin), (
                    f"Invalid pin state for {op}: pin and unpin cannot both be True."
                )

            return ops_list

        physical = []

        for idx, node in enumerate(logical_ops):
            name = node["op"]
            args = node.get("args", {})

            if name == OpName.SEM_FILTER:
                physical.append(
                    ops.SemFilter(
                        instruction=args["prompt"],
                        position=idx
                    )
                )

            elif name == OpName.SEM_MAP:
                physical.append(
                    ops.SemMap(
                        instruction=args["prompt"],
                        position=idx
                    )
                )
                MapRatioEstimator

            elif name in (OpName.SEM_CLASSIFY):
                physical.append(
                    ops.SemClassify(
                        classes=args['classes'],
                        position=idx
                    )
                )

            elif name == OpName.SEM_AGG:
                physical.append(
                    ops.SemAgg(
                        instruction=args["instruction"],
                        position=idx
                    )
                )

            elif name == OpName.SEM_TOPK:
                physical.append(
                    ops.SemTopK(
                        instruction=args["instruction"],
                        k=args["k"],
                        position=idx
                    )
                )

            elif name == OpName.JOIN:
                # Expand to CP + SemFilter
                physical.append(
                    ops.CartesianProduct(
                        right_table=list(data._data_source(None, args["right_table"], None)),
                        position=idx
                    )
                )
                physical.append(
                    ops.SemFilter(
                        instruction=args["instruction"],
                        position=idx
                    )
                )

            else:
                raise ValueError(f"Unknown op: {name}")

        apply_pin_unpin(physical)

        return tuple(physical)


    
    async def warmup(self, ctxs, plan):
        #sample
        out, stage_stat_list = await self.plan_executor.execute(ctxs[:int(len(ctxs)* 0.1)], plan)
        kv_manager = KVMemoryManager.get_instance()

        _max_len = -1
        for ctx in ctxs:
            token_len = kv_manager.token_length(ctx.input.data)
            _max_len = token_len if token_len > _max_len else _max_len

        for stages in stage_stat_list:
            sorted_items = sorted(stages.items(), key=lambda x: float(x[0]))

            sum_allocation = 0
            a = 10
            for i, (s, inout) in enumerate(sorted_items):
                ratio = inout['input'] / inout['output']

                if i == len(sorted_items) - 1:
                    allocation = kv_manager.capacity() - sum_allocation
                else:
                    allocation = (math.ceil(ratio) + a) * _max_len * kv_manager.bytes_per_token
                    sum_allocation += allocation

                print(s, ratio)
                kv_manager.register_stage(s, allocation)
                    
        pass
        

    async def execute(self, raw_request, query: Query):
        
        ctxs = list(data._data_source(raw_request, query.data_path, self.executor))
        MapRatioEstimator.instance(int(len(ctxs) * 0.1))
        physical_ops = self.parse_ops(query.ops)
        plan = self.build(ctxs, physical_ops)
        self.print_plan(plan)
                
        out = await self.warmup(ctxs, plan)
        out, _ = await self.plan_executor.execute(ctxs, plan)
        MapRatioEstimator.instance().reset()
        return out 
