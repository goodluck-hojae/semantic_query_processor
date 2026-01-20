
import asyncio


class SemanticChain:
    def __init__(self, ctx, *ops, bytes_per_token: int):
        self.ctx = ctx
        self.ops = ops
        self.bytes_per_token = bytes_per_token
        self.budget = self.estimate_token_budget(ctx.input.token_len)
        

    # budget function should be updated based on operations
    def estimate_token_budget(self, prompt_token_len) -> int:
        total_tokens = 0

        for op in self.ops:
            if not hasattr(op, "max_len"):
                raise AttributeError(
                    f"{op} must define `max_len`"
                )
            total_tokens += op.max_len

        self.budget = (prompt_token_len + total_tokens) * self.bytes_per_token
        return self.budget


    async def __call__(self):
        next = True
        for idx, op in enumerate(self.ops):
            if next:    
                next = await op(self.ctx)
