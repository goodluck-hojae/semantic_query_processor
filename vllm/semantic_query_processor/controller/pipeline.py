
class SemanticPipeline:
    def __init__(self, ctx, *ops, bytes_per_token: int):
        self.ctx = ctx
        self.ops = ops
        self.bytes_per_token = bytes_per_token
        self.budget = self.estimate_token_budget(ctx)
        
        
    def expand(self):
        """
        Lazily expand context if a CartesianProduct exists.
        Produces 1..N contexts.
        No async. No LLM calls.
        """

        ctx = self.ctx
        join_index = None

        # Locate join operator
        for i, op in enumerate(self.ops):
            if op.__class__.__name__ == "CartesianProduct":
                join_index = i
                break

        # No join → yield single ctx
        if join_index is None:
            yield ctx
            return

        join_op = self.ops[join_index]

        # Apply pure transforms before join (if any)
        for op in self.ops[:join_index]:
            if hasattr(op, "transform"):
                ctx = op.transform(ctx)

        # Lazy expansion
        for right in join_op.right_table:
            pair_ctx = ctx.clone()
            pair_ctx.right = right
            yield pair_ctx


    # budget function should be updated based on operations
    def estimate_token_budget(self, ctx) -> int:
        max_boundary = -1
        for op in self.ops:
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
                next = await op(self.ctx)
