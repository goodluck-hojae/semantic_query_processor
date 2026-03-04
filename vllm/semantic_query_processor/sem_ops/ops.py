from .base import BaseOp, OpKind, OpName
from vllm.semantic_query_processor.context import SemContext, SemanticInput, ExecutionState
from vllm.semantic_query_processor.budget import KVMemoryManager
from vllm.semantic_query_processor.execution.pipeline_execution import BlockingExecutor
from .prompt_utils import get_prompt, get_data_prompt
from typing import List


class SemFilter(BaseOp):

    TRUE = 'true'

    def __init__(self, instruction, pin=False, unpin=False, max_tokens=64):
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.instruction = instruction
        self.pin = pin
        self.unpin = unpin
        self.max_tokens = max_tokens


    def _build_data_prompt(self, ctx):
        if ctx.input.data is not None:
            return get_data_prompt(ctx.input.data)
        return get_data_prompt(ctx.input.left_input, ctx.input.right_input)


    def _build_full_prompt(self, ctx):
        if ctx.input.data is not None:
            return get_prompt(self.instruction, ctx.input.data, op=OpName.SEM_FILTER)
        return get_prompt(
            self.instruction,
            ctx.input.left_input,
            ctx.input.right_input,
            op=OpName.SEM_JOIN,
        )


    def _build_prompts(self, ctx):
        data_prompt = self._build_data_prompt(ctx)
        full_prompt = self._build_full_prompt(ctx)
        return data_prompt, full_prompt


    def estimate_tokens(self, ctx):
        _, prompt = self._build_prompts(ctx)
            
        prompt_str = KVMemoryManager.get_instance().tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

        return prompt_token_len + self.max_tokens


    async def __call__(self, ctx: SemContext):
        data_part, prompt = self._build_prompts(ctx)

        # Data part is only required to bin
        data_result = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=data_part,
                max_tokens=1,
                pin=self.pin,
        )

        result = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=prompt,
                max_tokens=self.max_tokens,
                pin=False,
        )
        verdict = result.text.strip().lower()

        if SemFilter.TRUE in verdict:
            passed = True
        else:
            passed = False
        
        ctx.output.append({
            str(self.__class__): verdict
        })
        
        if self.unpin and ctx.state.pin_req_id is not None:
            await ctx.state.executor.unpin(
                ctx.state.raw_request,
                ctx.state.pin_req_id,
            )
            ctx.state.pin_req_id = None

        if passed and self.pin:
            ctx.state.pin_req_id = data_result.request_id
        elif self.pin and data_result.request_id is not None:
            # Data part was pinned for this request, release it if filter failed.
            await ctx.state.executor.unpin(
                ctx.state.raw_request,
                data_result.request_id,
            )

        return passed
    

class SemMap(BaseOp):
    def __init__(
        self,
        instruction,
        max_tokens=256,
        expand=False,
        pin=False,
        unpin=False
    ):
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.expand = expand
        self.pin = pin
        self.unpin = unpin
        self.instruction_token_len = KVMemoryManager.get_instance().token_length(self.instruction) + max_tokens

        self.kind = OpKind.TUPLE_INDEPENDENT
            

    def _build_prompt(self, ctx):
        if ctx.input.data is None:
            raise ValueError("SemMap requires ctx.input.data and does not support left/right inputs.")
        return get_prompt(self.instruction, ctx.input.data, op=OpName.SEM_MAP)
    

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
        
        executor = ctx.state.executor
        raw_request = ctx.state.raw_request  

        prompt = self._build_prompt(ctx)

        output = await executor.execute(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=self.max_tokens,
            pin=self.pin,
        )
        prompt_str = KVMemoryManager.get_instance().tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=False,
        )
        appended_text = f"{prompt_str}{output.text}"
        ctx.input = SemanticInput(
            data=appended_text,
            token_len=KVMemoryManager.get_instance().token_length(appended_text),
        )
        ctx.output.append({
            str(self.__class__): output.text
        })

        if self.pin:
            ctx.state.pin_req_id = output.request_id
        elif self.unpin and ctx.state.pin_req_id:
            await executor.unpin(
                raw_request,
                ctx.state.pin_req_id,
            )
            ctx.state.pin_req_id = None

        return ctx

 
class CartesianProduct(BaseOp):
    def __init__(self, right_table):
        self.kind = OpKind.JOIN
        self.right_table = right_table

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

        
class SemClassify(BaseOp):
    def __init__(self, classes, pin=False, unpin=False):   
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.classes = list(classes)
        self.pin = pin
        self.unpin = unpin
        self.max_tokens = max(KVMemoryManager.get_instance().token_length(g) for g in self.classes) + 1

        self.instruction = "\n\n" \
                + "Choose exactly one class from the list below.\n" \
                + f"Class: {', '.join(self.classes)}\n" \
                + "Answer with the class name only:"
        self.instruction_token_len = KVMemoryManager.get_instance().token_length(self.instruction) + self.max_tokens + 1


    def _build_data_prompt(self, ctx):
        return get_data_prompt(ctx.input.data)


    def _build_full_prompt(self, ctx):
        return get_prompt(self.instruction, ctx.input.data, op=OpName.SEM_CLASSIFY)


    def _build_prompts(self, ctx):
        data_prompt = self._build_data_prompt(ctx)
        full_prompt = self._build_full_prompt(ctx)
        return data_prompt, full_prompt


    def estimate_tokens(self, ctx):
        _, prompt = self._build_prompts(ctx)
            
        prompt_str = KVMemoryManager.get_instance().tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

        return prompt_token_len + self.max_tokens


    async def __call__(self, ctx: SemContext) -> SemContext: 
 
        data_part, prompt = self._build_prompts(ctx)

        data_result = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=data_part,
                max_tokens=1,
                pin=self.pin,
        )

        result = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=prompt,
                max_tokens=self.max_tokens,
                pin=False,
        )
        group_result = result.text.strip().lower()
        group = ""
        for g in self.classes:
            if g.lower() in group_result:
                group = g
                break 

        ctx.output.append({
            str(self.__class__): str(group)
        })

        if self.pin:
            ctx.state.pin_req_id = data_result.request_id

        elif self.unpin and ctx.state.pin_req_id is not None:
            await ctx.state.executor.unpin(ctx.state.raw_request, ctx.state.pin_req_id)
            ctx.state.pin_req_id = None

        return ctx


class SemAgg(BaseOp):
    def __init__(self, instruction: str, max_tokens: int = 8192, concurrency: int = 8):
        self.kind = OpKind.BLOCKING
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.concurrency = concurrency


    async def __call__(self, ctxs: List[SemContext]) -> List[SemContext]:
        working_set = list(ctxs)
        if not working_set:
            return []

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
        instruction_overhead = KVMemoryManager.get_instance().token_length(
            "\n\n" + self.instruction + "\n\n"
        )
        per_doc_overhead = KVMemoryManager.get_instance().token_length(
            "\n\nDocument 1:\n"
        )
        cur_tokens = instruction_overhead

        for ctx in ctxs:
            t = ctx.input.token_len
            item_tokens = t + per_doc_overhead
            # Allow an oversized single item as its own chunk instead of failing early.
            if cur and cur_tokens + item_tokens > self.max_tokens:
                chunks.append(cur)
                cur = []
                cur_tokens = instruction_overhead

            cur.append(ctx)
            cur_tokens += item_tokens

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
