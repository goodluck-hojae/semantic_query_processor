from vllm.semantic_query_processor.sem_ops import ops
from vllm.semantic_query_processor.budget import KVMemoryManager

def pipeline_builder(ops, stage_id):
    def _pipeline(ctx):
        return SemanticPipeline(
            ctx,
            *ops,
            bytes_per_token=KVMemoryManager.get_instance().bytes_per_token,
            stage_id=stage_id,
        )
    _pipeline.ops = ops
    _pipeline.stage_id = stage_id
    return _pipeline



class SemanticPipeline:
    def __init__(self, ctx, *ops, bytes_per_token: int, stage_id: str):
        self.ctx = ctx
        self.ops = ops
        self.bytes_per_token = bytes_per_token
        self.stage_id = stage_id
        self.budget = self.estimate_token_budget(ctx)
        

    # budget function should be updated based on operations
    def estimate_token_budget(self, ctx) -> int:
        max_boundary = -1
        for op in self.ops:
            if isinstance(op, ops.CartesianProduct):
                break

            estimated_tokens = op.estimate_tokens(ctx)
            if not hasattr(op, "estimate_tokens"):
                raise AttributeError(
                    f"{op} must define `estimate_tokens`"
                )
            if estimated_tokens > max_boundary:
                max_boundary = estimated_tokens

        self.budget = max_boundary * self.bytes_per_token
        return self.budget


    async def __call__(self):
        next = True
        for idx, op in enumerate(self.ops):
            if next:
                if isinstance(op, ops.CartesianProduct):
                    ctxs = op(self.ctx)
                    pipelines = []
                    for ctx in ctxs:
                        pipeline = pipeline_builder(self.ops[idx+1:], self.stage_id)(ctx)
                        pipelines.append(pipeline)
                    return pipelines
                next = await op(self.ctx)
        return None
