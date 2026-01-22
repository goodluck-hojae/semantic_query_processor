from .base import BaseOp, OpKind
from vllm.semantic_query_processor.context import SemContext, SemanticInput, ExecutionState
from .endpoint import unpin_request, completion_call_internal
from typing import List

class SemFilter(BaseOp):
    def __init__(self, instruction, pin=False, unpin=False, max_len=10):   
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.instruction = instruction
        self.pin = pin
        self.unpin = unpin
        self.max_len = max_len

    async def __call__(self, ctx):
        prompt = (
            ctx.input.data
            + "\n\n"
            + self.instruction
            + "\n\n"
            + "Answer True or False. Answer:"
        )

        res, req = await completion_call_internal(
            ctx.state.raw_request,
            prompt,
            self.max_len,
            pin=self.pin
        )

        # if the answer is no, we can evict the pinned kv immediately
        if ctx.state.pin_req_id is not None and self.unpin:
            engine = ctx.state.raw_request.app.state.engine_client
            await unpin_request(engine, ctx.state.pin_req_id)
            ctx.state.pin_req_id = None

        if self.pin:
            ctx.state.pin_req_id = res["id"]

        return True
    

class SemMap(BaseOp):
    def __init__(self, instruction, pin=False, max_len=128):   
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.instruction = instruction
        self.pin = pin
        self.max_len = max_len

    async def __call__(self, ctx):
        prompt = (
            ctx.input.data
            + "\n\n"
            + self.instruction
            + "\n\n"
        )

        res, req = await completion_call_internal(
            ctx.state.raw_request,
            prompt,
            self.max_len,
            pin=self.pin
        )

        # sem_map always unpins the previous pin_req_id 
        if ctx.state.pin_req_id is not None: 
            engine = ctx.state.raw_request.app.state.engine_client
            await unpin_request(engine, ctx.state.pin_req_id)

        if self.pin:
            # TODO impl pin gen request if needed and update budget
            # Create new request for pinning res["choices"][0]["text"]
            # Udpate budget
            pass
        return res["choices"][0]["text"]


        
    

class SemGroupBy(BaseOp):
    def __init__(self, groups, pin=False, unpin=False, max_len=20):   
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.groups = groups
        self.pin = pin
        self.unpin = unpin
        self.max_len = max_len

    async def __call__(self, ctx):
        prompt = (
            ctx.input.data
            + "\n\n"
            + f'Groups: [{str(self.groups)}]'
            + "\n\n"
            + "Choose the closest group. Answer:"
        )

        res, req = await completion_call_internal(
            ctx.state.raw_request,
            prompt,
            self.max_len,
            pin=self.pin
        )
        ctx.state.pin_req_id = res["id"]

        if ctx.state.pin_req_id is not None and self.unpin:
            engine = ctx.raw_request.app.state.engine_client
            await engine.engine_core.call_utility_async(
                "unpin_kv",
                ctx.state.pin_req_id,
            )

        return True


class SemJoin(BaseOp):
    def __init__(self, right_table):
        self.kind = OpKind.BLOCKING
        self.right_table = right_table

    async def __call__(self, ctxs):
        out = []

        for ctx in ctxs:
            for right in self.right_table:

                new_ctx = SemContext(
                    input=SemanticInput(
                        data=ctx.input.data + "\n\n" + str(right.input.data),
                        token_len=ctx.input.token_len * 2,  #  TODO
                    ),
                    state=ExecutionState(
                        raw_request=ctx.state.raw_request,
                        pin_req_id=None,
                    ),
                )
                out.append(new_ctx)

        return out


class SemAgg(BaseOp):

    def __init__(self, instruction: str, max_tokens: int):
        self.kind = OpKind.BLOCKING
        self.instruction = instruction
        self.max_tokens = max_tokens


    async def __call__(self, ctxs: List[SemContext]) -> List[SemContext]:
        aggregate_set = ctxs

        while len(aggregate_set) > 1:
            next_aggregate_set: List[SemContext] = []
            chunks = self._chunk_by_tokens(aggregate_set)

            for chunk in chunks:
                if len(chunk) == 1:
                    next_aggregate_set.append(chunk[0])
                    continue

                merged = await self._reduce_chunk(chunk)
                next_aggregate_set.append(merged)

            aggregate_set = next_aggregate_set

        return aggregate_set

    def _chunk_by_tokens(self, ctxs: List[SemContext]) -> List[List[SemContext]]:
        chunks = []
        cur = []
        cur_tokens = 0

        for ctx in ctxs:
            t = ctx.input.token_len
            if t > self.max_tokens:
                raise RuntimeError("Single context exceeds max_tokens")

            if cur_tokens + t > self.max_tokens:
                chunks.append(cur)
                cur = []
                cur_tokens = 0

            cur.append(ctx)
            cur_tokens += t

        if cur:
            chunks.append(cur)

        return chunks

    async def _reduce_chunk(self, chunk: List[SemContext]) -> SemContext:
        """
        Reduce N contexts into 1 via LLM.
        """

        raw_request = chunk[0].state.raw_request

        prompt = ""
        total_tokens = 0

        for i, ctx in enumerate(chunk, 1):
            prompt += f"\n\nDocument {i}:\n{ctx.input.data}"
            total_tokens += ctx.input.token_len

        prompt += "\n\n" + self.instruction + "\n\n"

        res, _ = await completion_call_internal(
            raw_request,
            prompt,
            self.max_tokens,
        )

        text = res["choices"][0]["text"]

        return SemContext(
            input=SemanticInput(
                data=text,
                token_len=max(1, total_tokens // 2),  # conservative estimate
            ),
            state=ExecutionState(
                raw_request=raw_request,
                pin_req_id=None,
            ),
        )


class SemTopK(BaseOp):
    def __init__(self, temp, pin=False, unpin=False, max_len=20):   
        self.kind = OpKind.TUPLE_DEPENDENT
        self.temp = temp
        self.pin = pin
        self.unpin = unpin
        self.max_len = max_len

    async def __call__(self, ctx):
        print('SemTopK called')
        return self.temp