from .base import BaseOp, OpKind
from vllm.semantic_query_processor.context import SemContext, SemanticInput, ExecutionState
from vllm.semantic_query_processor.budget import KVMemoryManager
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
    def __init__(self, instruction: str, max_tokens: int, concurrency: int = 8):
        self.kind = OpKind.BLOCKING
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.concurrency = concurrency


    async def __call__(self, ctxs: List[SemContext]) -> List[SemContext]:
        working_set = list(ctxs)

        while len(working_set) > 1:
            chunks = self._chunk_by_tokens(working_set)

            # singletons pass through
            passthrough = [c[0] for c in chunks if len(c) == 1]
            reducible = [c for c in chunks if len(c) > 1]

            if not reducible:
                return passthrough

            reduced = await KVMemoryManager.get_instance().execute_tasks(
                seeds=reducible,
                task_builder=self._build_reducer,
                concurrency=self.concurrency,
            )

            working_set = passthrough + reduced

        return working_set


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


    def _build_reducer(self, chunk: List[SemContext]):
        parent = self
        kv_budget = KVMemoryManager.get_instance()

        class Reducer:
            def __init__(self, chunk: List[SemContext]):
                self.chunk = chunk

                prompt = ""
                total_tokens = 0
                for i, ctx in enumerate(chunk, 1):
                    prompt += f"\n\nDocument {i}:\n{ctx.input.data}"
                    total_tokens += ctx.input.token_len

                prompt += "\n\n" + parent.instruction + "\n\n"

                prompt_token_len = kv_budget.token_length(prompt)

                self.budget = (
                    (prompt_token_len + total_tokens)
                    * kv_budget.bytes_per_token
                )

            async def __call__(self) -> SemContext:
                return await parent._reduce_chunk(self.chunk)

        return Reducer(chunk)


    async def _reduce_chunk(self, chunk: List[SemContext]) -> SemContext:
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
                token_len=max(1, total_tokens // 2),
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