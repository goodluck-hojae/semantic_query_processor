from .base import BaseOp, OpKind
from vllm.semantic_query_processor.context import SemContext, SemanticInput, ExecutionState
from vllm.semantic_query_processor.budget import KVMemoryManager
from typing import List


class SemFilter(BaseOp):
    def __init__(self, instruction, pin=False, unpin=False, max_tokens=5):
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.instruction = instruction
        self.pin = pin
        self.unpin = unpin
        self.max_tokens = max_tokens

    async def __call__(self, ctx: SemContext):
        prompt = (
            ctx.input.data
            + "\n\n"
            + self.instruction
            + "\n\n"
            + "Answer True or False. Answer:"
        )

        result = await ctx.state.executor.complete(
            raw_request=ctx.state.raw_request,
            prompt=prompt,
            max_tokens=self.max_tokens,
            pin=self.pin,
        )


        verdict = result.text.strip().lower()

        if "true" in verdict:
            passed = True
        elif "false" in verdict:
            passed = False
        else:
            raise ValueError(f"Invalid filter output: {verdict}")

        if (not passed or self.unpin) and ctx.state.pin_req_id is not None:
            await ctx.state.executor.unpin(
                ctx.state.raw_request,
                ctx.state.pin_req_id,
            )
            ctx.state.pin_req_id = None

        elif self.pin:
            ctx.state.pin_req_id = result.request_id

        return passed
    

class SemMap(BaseOp):
    """
    Case 1: use_output_as_prompt = False
        - Retain input prompt KV cache and output context KV cache 
        - Downstream prompt uses input (prompt + output context)
        - pin output context KV cache blocks
        - TUPLE_INDEPENDENT

    Case 2: use_output_as_prompt = True AND max_tokens <= input_prompt_len
        - Output replaces prompt
        - Output length cannot exceed input
        - No rebudgeting required, conservatively, it maintains existing budget
        - TUPLE_INDEPENDENT

    Case 3: use_output_as_prompt = True AND max_tokens > input_prompt_len
        - Output replaces prompt
        - Output exceeds input
        - Requires rebudgeting, so plays as a blocking operator
        - BLOCKING
    """

    def __init__(
        self,
        instruction,
        max_tokens=256,
        use_output_as_prompt: bool = False,
        expand=False,
        pin: bool = False,
        concurrency: int = 8,
    ):
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.use_output_as_prompt = use_output_as_prompt
        self.expand = expand
        self.pin = pin
        self.concurrency = concurrency

        if use_output_as_prompt and expand:
            self.kind = OpKind.BLOCKING
        else:
            self.kind = OpKind.TUPLE_INDEPENDENT

    async def __call__(self, arg):
        if self.kind == OpKind.BLOCKING:
            # List[SemContext]
            return await self._call_blocking(arg)
        else:
            # SemContext
            return await self._call_tuple(arg)


    async def _call_tuple(self, ctx: SemContext) -> SemContext:
        executor = ctx.state.executor
        raw_request = ctx.state.raw_request

        prompt = (
            ctx.input.data
            + "\n\n"
            + self.instruction
            + "\n\n"
        )

        output = await executor.complete(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=self.max_tokens,
            pin=False,
        )

        # Case 1
        if not self.use_output_as_prompt:
            downstream_prompt = prompt + output.text
            generated = await executor.complete(
                raw_request=raw_request,
                prompt=downstream_prompt,
                max_tokens=1,
                pin=self.pin,
            )

            if self.pin:
                ctx.state.pin_req_id = generated.request_id

            ctx.input = SemanticInput(
                data=downstream_prompt,
                token_len=KVMemoryManager.get_instance().token_length(downstream_prompt),
            )
            return ctx
        else:
            # -------- Case 2 / Case 3 --------
            if ctx.state.pin_req_id is not None:
                await executor.unpin(
                    raw_request,
                    ctx.state.pin_req_id,
                )
                ctx.state.pin_req_id = None

            outcome_req = await executor.complete(
                raw_request=raw_request,
                prompt=output.text,
                max_tokens=1,
                pin=self.pin,
            )

            if self.pin:
                ctx.state.pin_req_id = outcome_req.request_id

            ctx.input = SemanticInput(
                data=output.text,
                token_len=KVMemoryManager.get_instance().token_length(output.text),
            )

            return ctx


    # Blocking (Case 3)
    async def _call_blocking(self, ctxs):
        parent = self

        def task_builder(ctx: SemContext):
            class MapTask:
                def __init__(self, ctx):
                    self.ctx = ctx
                    self.budget = parent.max_tokens * KVMemoryManager.get_instance().bytes_per_token

                async def __call__(self):
                    return await parent._call_tuple(self.ctx)

            return MapTask(ctx)

        return await KVMemoryManager.get_instance().execute_tasks(
            seeds=ctxs,
            task_builder=task_builder,
            concurrency=self.concurrency,
        )


class SemGroupBy(BaseOp):
    def __init__(self, groups, pin=False, unpin=False):   
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.groups = list(groups)
        self.pin = pin
        self.unpin = unpin
        self.max_tokens = max(KVMemoryManager.get_instance().token_length(g) for g in self.groups) + 1

    async def __call__(self, ctx: SemContext) -> SemContext:
        executor = ctx.state.executor
        raw_request = ctx.state.raw_request

        prompt = (
            ctx.input.data
            + "\n\n"
            + "Choose exactly one group from the list below.\n"
            + f"Groups: {', '.join(self.groups)}\n"
            + "Answer with the group name only:"
        )

        result = await executor.complete(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=self.max_tokens,
            pin=self.pin,
        )

        group = result.text.strip()

        if group not in self.groups:
            raise ValueError(f"Invalid group '{group}', expected one of {self.groups}")

        ctx.group_key = group

        if self.pin:
            ctx.state.pin_req_id = result.request_id

        if self.unpin and ctx.state.pin_req_id is not None:
            await executor.unpin(raw_request, ctx.state.pin_req_id)
            ctx.state.pin_req_id = None

        return ctx


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
    def __init__(self, instruction: str, max_tokens: int = 8192, concurrency: int = 8):
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
            # TODO here is a bug
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
        executor = chunk[0].state.executor
        raw_request = chunk[0].state.raw_request

        prompt = ""
        total_tokens = 0

        for i, ctx in enumerate(chunk, 1):
            prompt += f"\n\nDocument {i}:\n{ctx.input.data}"
            total_tokens += ctx.input.token_len

        prompt += "\n\n" + self.instruction + "\n\n"

        result = await executor.complete(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=self.max_tokens,
            pin=False,
        )

        text = result.text

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
    def __init__(self, instruction: str, k: int= 10, concurrency: int = 20):
        self.kind = OpKind.BLOCKING
        self.instruction = instruction
        self.k = k
        self.max_tokens = 5
        self.concurrency = concurrency

    async def __call__(self, ctxs: List[SemContext]) -> List[SemContext]:
        if len(ctxs) <= self.k:
            return await self._rank(ctxs)

        topk = await self._quickselect(ctxs, self.k)
        return await self._rank(topk)


    async def _quickselect(self, ctxs: List[SemContext], k: int) -> List[SemContext]:
        if len(ctxs) <= k:
            return ctxs

        pivot = ctxs[0]
        others = ctxs[1:]

        better, worse = await self._partition(pivot, others)

        if len(better) >= k:
            return await self._quickselect(better, k)
        else:
            return better + await self._quickselect(worse, k - len(better))


    async def _partition(self, pivot: SemContext, others: List[SemContext]):
        executor = pivot.state.executor

        def build_task(other: SemContext):
            raw_request = pivot.state.raw_request
            instruction = self.instruction
            max_tokens = self.max_tokens

            class CompareTask:
                def __init__(self):
                    self.prompt = (
                        f"Document A:\n{pivot.input.data}\n\n"
                        f"Document B:\n{other.input.data}\n\n"
                        f"{instruction}\n"
                        f"Answer with 'A' or 'B'.\n\nAnswer:"
                    )
                    tokens = KVMemoryManager.get_instance().token_length(self.prompt)
                    self.budget = tokens * KVMemoryManager.get_instance().bytes_per_token


                async def __call__(self):
                    result = await executor.complete(
                        raw_request=raw_request,
                        prompt=self.prompt,
                        max_tokens=max_tokens,
                        pin=False,
                    )
                    return other, result.text.strip().upper()
                

            return CompareTask()

        results = await KVMemoryManager.get_instance().execute_tasks(
            seeds=others,
            task_builder=build_task,
            concurrency=self.concurrency,
        )

        better, worse = [], []
        for other, verdict in results:
            if verdict.startswith("B"):
                better.append(other)
            else:
                worse.append(other)

        return better, worse


    async def _rank(self, ctxs: List[SemContext]) -> List[SemContext]:
        if len(ctxs) <= 1:
            return ctxs

        ranked: List[SemContext] = []

        for ctx in ctxs:
            inserted = False
            for i, other in enumerate(ranked):
                better = await self._compare(ctx, other)
                if better:
                    ranked.insert(i, ctx)
                    inserted = True
                    break
            if not inserted:
                ranked.append(ctx)

        return ranked

    async def _compare(self, a: SemContext, b: SemContext) -> bool:
        executor = a.state.executor 
        raw_request = a.state.raw_request
        prompt = (
            f"Document A:\n{a.input.data}\n\n"
            f"Document B:\n{b.input.data}\n\n"
            f"{self.instruction}\n"
            f"Answer with 'A' or 'B'.\n\nAnswer:"
        )

        result = await executor.complete(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=self.max_tokens,
            pin=False,
        )
        return result.text.strip().upper().startswith("A")