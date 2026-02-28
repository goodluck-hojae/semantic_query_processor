from .base import BaseOp, OpKind
from vllm.semantic_query_processor.context import SemContext, SemanticInput, ExecutionState
from vllm.semantic_query_processor.budget import KVMemoryManager
from vllm.semantic_query_processor.execution.pipeline_execution import BlockingExecutor
from .prompt_utils import get_prompt
from typing import List
import json


class SemFilter(BaseOp):
    def __init__(self, instruction, pin=False, unpin=False, max_tokens=64):
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.instruction = instruction
        self.pin = pin
        self.unpin = unpin
        self.max_tokens = max_tokens


    def _build_prompt(self, ctx):
        
        if ctx.input.data is not None:
            prompt = get_prompt(self.instruction, ctx.input.data, op='sem_filter')
        else:
            prompt = get_prompt(self.instruction, ctx.input.left_input, ctx.input.right_input, op='sem_join')


        return prompt


    def estimate_tokens(self, ctx):
        prompt = self._build_prompt(ctx)
            
        prompt_str = KVMemoryManager.get_instance().tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

        return prompt_token_len + self.max_tokens


    async def __call__(self, ctx: SemContext):
        prompt = self._build_prompt(ctx)

        result = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=prompt,
                max_tokens=self.max_tokens,
                pin=self.pin,
        )
        verdict = result.text.strip().lower()

        if "true" in verdict:
            passed = True
        else:
            passed = False
            ctx.state.predicate = False
        
        ctx.output.append({
            str(self.__class__): verdict
        })
        
        if (not passed or self.unpin) and ctx.state.pin_req_id:
            await ctx.state.executor.unpin(
                ctx.state.raw_request,
                ctx.state.pin_req_id,
            )
            ctx.state.pin_req_id = None
        elif (not passed or self.pin) and result.request_id is not None:
            await ctx.state.executor.unpin(
                ctx.state.raw_request,
                result.request_id,
            )
        elif self.pin:
            ctx.state.pin_req_id = result.request_id

        return passed
    


class SemMap(BaseOp):
    """
    Case 1: chain_of_thought = True
        - Retain input prompt KV cache and output context KV cache 
        - Downstream prompt uses input (prompt + output context)
        - pin output context KV cache blocks
        - TUPLE_INDEPENDENT

    Case 2: chain_of_thought = False AND max_tokens <= input_prompt_len
        - Output replaces prompt
        - Output length cannot exceed input
        - No rebudgeting required, conservatively, it maintains existing budget
        - TUPLE_INDEPENDENT

    Case 3: chain_of_thought = False AND max_tokens > input_prompt_len
        - Output replaces prompt
        - Output exceeds input
        - Requires rebudgeting, so plays as a blocking operator
        - BLOCKING
    """

    def __init__(
        self,
        instruction,
        max_tokens=256,
        chain_of_thought: bool = True,
        expand=False,
        pin=False,
        unpin=False
    ):
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.chain_of_thought = chain_of_thought
        self.expand = expand
        self.pin = pin
        self.unpin = unpin
        self.instruction_token_len = KVMemoryManager.get_instance().token_length(self.instruction) + max_tokens

        if chain_of_thought and expand:
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


    def _build_prompt(self, ctx):
        if ctx.input.data is not None:
            prompt = get_prompt(self.instruction, ctx.input.data, op='sem_map')
        else:
            prompt = get_prompt(self.instruction, ctx.input.left_input, ctx.input.right_input, op='sem_map')

        return prompt
    

    def estimate_tokens(self, ctx):
        prompt = self._build_prompt(ctx)
        prompt_str = KVMemoryManager.get_instance().tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)
        return prompt_token_len + self.max_tokens


    async def _call_tuple(self, ctx: SemContext) -> SemContext:
        executor = ctx.state.executor
        raw_request = ctx.state.raw_request

        prompt = self._build_prompt(ctx)


        # Case 1 (Chain of thoughts)
        if self.chain_of_thought:
            output = await ctx.state.executor.execute(
                raw_request=raw_request,
                prompt=prompt,
                max_tokens=self.max_tokens,
                pin=self.pin,
            )
            ctx.output.append({
                str(self.__class__): output.text
            })
        
            if self.pin:
                ctx.state.pin_req_id = output.request_id

            if self.unpin and ctx.state.pin_req_id:
                await executor.unpin(
                    raw_request,
                    ctx.state.pin_req_id,
                )
                ctx.state.pin_req_id = None

            return ctx
        
        else:
            #TODO do this later
            # -------- Case 2 / Case 3 --------
            output = await llm_func(
                raw_request=raw_request,
                prompt=message,
                max_tokens=self.max_tokens,
                pin=False,
            )
            
            if ctx.state.pin_req_id is not None:
                await executor.unpin(
                    raw_request,
                    ctx.state.pin_req_id,
                )
                ctx.state.pin_req_id = None

            outcome_req = await llm_func(
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


class CartesianProduct(BaseOp):
    def __init__(self, right_table):
        self.kind = OpKind.JOIN
        self.right_table = right_table

    def _build_prompt(self, data, data2):

        prompt = get_prompt(self.instruction, data, data2, op='sem_join')
        return prompt


    def __call__(self, ctx):
        out = []

        for right in self.right_table:
            new_ctx = SemContext(
                input=SemanticInput(
                    left_input=ctx.input.data,
                    right_input=right.input.data,
                ),
                state=ExecutionState(
                    raw_request=ctx.state.raw_request,
                    pin_req_id=None,
                    executor=ctx.state.executor
                ),
                )
            out.append(new_ctx)
        return out

        


class SemGroupBy(BaseOp):
    def __init__(self, groups, pin=False, unpin=False):   
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.groups = list(groups)
        self.pin = pin
        self.unpin = unpin
        self.max_tokens = max(KVMemoryManager.get_instance().token_length(g) for g in self.groups) + 1

        self.instruction = "\n\n" \
                + "Choose exactly one group from the list below.\n" \
                + f"Groups: {', '.join(self.groups)}\n" \
                + "Answer with the group name only:"
        self.instruction_token_len = KVMemoryManager.get_instance().token_length(self.instruction) + self.max_tokens + 1


    def _build_prompt(self, ctx):
        if ctx.input.data is not None:
            prompt = get_prompt(self.instruction, ctx.input.data, op='sem_groupby')
        return prompt


    def estimate_tokens(self, ctx):
        prompt = self._build_prompt(ctx)
            
        prompt_str = KVMemoryManager.get_instance().tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

        return prompt_token_len + self.max_tokens


    async def __call__(self, ctx: SemContext) -> SemContext: 

        prompt = self._build_prompt(ctx)

        result = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=prompt,
                max_tokens=self.max_tokens,
                pin=self.pin,
        )
        group_result = result.text.strip().lower()
        group = ""
        for g in self.groups:
            if g.lower() in group_result:
                group = g
                break 

        ctx.output.append({
            str(self.__class__): str(group)
        })
        if self.pin:
            ctx.state.pin_req_id = result.request_id

        if self.unpin and ctx.state.pin_req_id is not None:
            await ctx.state.executor.unpin(ctx.state.raw_request, ctx.state.pin_req_id)
            ctx.state.pin_req_id = None

        return ctx


class SemAgg(BaseOp):
    def __init__(self, instruction: str, max_tokens: int = 2048, concurrency: int = 8):
        self.kind = OpKind.BLOCKING
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.concurrency = concurrency


    async def __call__(self, ctxs: List[SemContext]) -> List[SemContext]:
        working_set = list(ctxs)

        while len(working_set) > 1:
            chunks = self._chunk_by_tokens(working_set)

            reducible = [c for c in chunks if len(c) > 1]
            passthrough = [c[0] for c in chunks if len(c) == 1]

            if reducible:
                reduced = await BlockingExecutor.execute_tasks(
                    seeds=reducible,
                    task_builder=self._build_reducer,
                    concurrency=self.concurrency,
                )
                
                working_set = passthrough + reduced
            else:
                working_set = await BlockingExecutor.execute_tasks(
                    seeds=[working_set],
                    task_builder=self._build_reducer,
                    concurrency=1,
                )
                
        working_set[0].output.append({
            str(self.__class__): working_set[0].input.data
        })
        
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
        
        class Reducer:
            def __init__(self, chunk: List[SemContext]):
                self.chunk = chunk

                prompt = ""
                for i, ctx in enumerate(chunk, 1):
                    prompt += f"\n\nDocument {i}:\n{ctx.input.data}"

                prompt += "\n\n" + parent.instruction + "\n\n"

                prompt_token_len = KVMemoryManager.get_instance().token_length(prompt)

                self.budget = (prompt_token_len + parent.max_tokens) * KVMemoryManager.get_instance().bytes_per_token 

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

        ctx = SemContext(
            input=SemanticInput(
                data=result.text,
                token_len=KVMemoryManager.get_instance().token_length(result.text)
            ),
            state=ExecutionState(
                raw_request=raw_request,
                pin_req_id=None,
                executor=executor
            ),
        )
        return ctx


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

        results = await BlockingExecutor.execute_tasks(
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