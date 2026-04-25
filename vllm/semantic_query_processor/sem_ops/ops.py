from .base import BaseOp, OpKind, OpName
from vllm.semantic_query_processor.context import (
    RETRY_TASK,
    ExecutionState,
    SemContext,
    SemanticInput,
)
from vllm.semantic_query_processor.budget import KVMemoryManager
from vllm.semantic_query_processor.execution.pipeline_execution import BlockingExecutor
from vllm.semantic_query_processor.controller.map_estimator import MapRatioEstimator
from .prompt_utils import get_prompt, get_system_prompt, add_assistant_prompt
import requests
from typing import Any, List


class SemFilter(BaseOp):

    TRUE = 'true'
    FALSE = 'false'
    LOG = False

    def __init__(self, instruction, negate=False, pin=False, unpin=False, max_tokens=8, position=-1):
        super().__init__(kind=OpKind.TUPLE_INDEPENDENT, position=position)
        self.instruction = instruction
        self.negate = negate
        self.pin = pin
        self.unpin = unpin
        self.max_tokens = max_tokens

    def _build_prompts(self, ctx):
        if ctx.input.right_data:
            joined_data = ctx.input.data + ctx.input.right_data
            data_prompt = joined_data if 'system' in joined_data[0]['role'] else get_system_prompt() + joined_data
            full_prompt = get_prompt(self.instruction, joined_data, op=OpName.SEM_JOIN)
        else:
            data_prompt = ctx.input.data if 'system' in ctx.input.data[0]['role'] else get_system_prompt() + ctx.input.data
            full_prompt = get_prompt(self.instruction, ctx.input.data, op=OpName.SEM_FILTER)
        return data_prompt, full_prompt
    
    def estimate_tokens(self, ctx):
        _, prompt = self._build_prompts(ctx)
            
        prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

        return prompt_token_len + self.max_tokens


    async def __call__(self, ctx: SemContext, priority: int = 0):
        data_part, prompt = self._build_prompts(ctx)

        # Data part is only required to bin
        # data_result = await ctx.state.executor.execute(
        #         raw_request=ctx.state.raw_request,
        #         prompt=data_part,
        #         max_tokens=1,
        #         pin=self.pin,
        # )
        output = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=prompt,
                max_tokens=self.max_tokens,
                pin=self.pin,
                priority=priority,
        )
        if self.pin:
            ctx.state.pin_req_id = output.request_id

        # appended_prompt, appended_prompt_str = add_assistant_prompt(prompt, output.text)
        verdict = output.text.strip().lower()
        
        ctx.input.data = prompt[:-1]
        
        if bool(ctx.input.right_data):
            ctx.input.right_data = []

        if SemFilter.FALSE in verdict:
            passed = False
        else:
            passed = True

        # if SemFilter.TRUE in verdict:
        #     passed = True
        # else:
        #     passed = False

        if self.negate:
            passed = not passed
            
        ctx.output.append({
            str(self.__class__): verdict
        })


        if (self.unpin or not passed) and ctx.state.pin_req_id is not None:
            if self.LOG:
                print(
                    "[sem-op] "
                    f"SemFilter unpin pin_req_id={ctx.state.pin_req_id} "
                    f"passed={passed} "
                    f"self_unpin={self.unpin}"
                )
            await ctx.state.executor.unpin(ctx.state.raw_request, ctx.state.pin_req_id)
            ctx.state.pin_req_id = None

        return passed
    

class SemMap(BaseOp):
    MAX_TOKEN_LIMIT = 4096
    LOG = False
    LOG_RETRY_TRACE = True
    def __init__(
        self,
        instruction,
        max_tokens=MAX_TOKEN_LIMIT,
        expand=False,
        pin=False,
        unpin=False,
        position=-1
    ):
        super().__init__(kind=OpKind.TUPLE_INDEPENDENT, position=position)
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.expand = expand
        self.pin = pin
        self.unpin = unpin
        self.instruction_token_len = KVMemoryManager.get_instance().token_length(self.instruction) + max_tokens

    def _planned_max_tokens(self, ctx, prompt_token_len=None) -> int:
        if (
            ctx.state.retry_op_position == self.position
            and ctx.state.retry_max_tokens > 0
        ):
            return ctx.state.retry_max_tokens

        if prompt_token_len is None:
            prompt = self._build_prompt(ctx)
            prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
            prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

        ratio = MapRatioEstimator.instance().get_ratio(self.position)
        return int(ratio * prompt_token_len) if ratio else 1 #int(prompt_token_len)  #1

    def _build_prompt(self, ctx):
        return get_prompt(self.instruction, ctx.input.data, op=OpName.SEM_MAP)
    

    def estimate_tokens(self, ctx):
        prompt = self._build_prompt(ctx)
        prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)
        self.max_tokens = self._planned_max_tokens(ctx, prompt_token_len)
        return prompt_token_len + self.max_tokens
    
    
    async def __call__(self, ctx: SemContext, priority: int = 0):
        
        executor = ctx.state.executor
        raw_request = ctx.state.raw_request
        previous_pin_req_id = ctx.state.pin_req_id if self.pin else None

        prompt = self._build_prompt(ctx)
        prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
        input_token_len = KVMemoryManager.get_instance().token_length(prompt_str)
        max_tokens = self._planned_max_tokens(ctx, input_token_len)

        output = await executor.execute(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=max_tokens,
            pin=self.pin,
            priority=priority,
        )
        
        if output.finish_reason == "length" and max_tokens < SemMap.MAX_TOKEN_LIMIT:
            if self.pin:
                ctx.state.pin_req_id = output.request_id
            ctx.state.retry_op_position = self.position
            ctx.state.retry_max_tokens = SemMap.MAX_TOKEN_LIMIT
            if self.LOG_RETRY_TRACE:
                print(
                    "[sem-map] "
                    f"retry-trigger stage={ctx.state.stage_id} "
                    f"position={self.position} "
                    f"request_id={output.request_id} "
                    f"pin_req_id={ctx.state.pin_req_id} "
                    f"from_max_tokens={max_tokens} "
                    f"to_max_tokens={ctx.state.retry_max_tokens} "
                    f"input_tokens={input_token_len}"
                )
            if self.LOG:
                print(
                    "[sem-map] "
                    f"retry-requeue stage={ctx.state.stage_id} "
                    f"position={self.position} "
                    f"request_id={output.request_id} "
                    f"from_max_tokens={max_tokens} "
                    f"to_max_tokens={SemMap.MAX_TOKEN_LIMIT}"
                )
            return RETRY_TASK

        if ctx.state.retry_op_position == self.position:
            ctx.state.retry_op_position = -1
            ctx.state.retry_max_tokens = 0

        appended_prompt, appended_prompt_str = add_assistant_prompt(prompt, output.text)
        ctx.input.data = appended_prompt
        # Update ratio
        MapRatioEstimator.instance().update(self.position, input_token_len, KVMemoryManager.get_instance().token_length(appended_prompt_str)-input_token_len)

        ctx.output.append({
            str(self.__class__): output.text
        })

        if self.pin:
            if (
                previous_pin_req_id is not None
                and previous_pin_req_id != output.request_id
            ):
                await executor.unpin(raw_request, previous_pin_req_id)
            ctx.state.pin_req_id = output.request_id
        elif self.unpin and ctx.state.pin_req_id:
            if self.LOG:
                print(
                    "[sem-op] "
                    f"SemMap unpin pin_req_id={ctx.state.pin_req_id}"
                )
            await executor.unpin(raw_request, ctx.state.pin_req_id)
            ctx.state.pin_req_id = None

        return ctx

 
class CartesianProduct(BaseOp):
    def __init__(self, right_table, position=-1):
        super().__init__(kind=OpKind.JOIN, position=position)
        self.right_table = right_table

    def __call__(self, ctx):
        out = []

        for right in self.right_table:
            base_input = SemanticInput(
                data=ctx.input.data[:],
                right_data=[]
            )
            _input = base_input.add_right(right.input.data)
            new_ctx = SemContext(
                input=_input,
                state=ExecutionState(
                    raw_request=ctx.state.raw_request,
                    pin_req_id=None,
                    executor=ctx.state.executor
                ),
                )
            out.append(new_ctx)
        return out


class IndexedCartesianProduct(BaseOp):
    def __init__(
        self,
        right_table,
        service_address="127.0.0.1",
        service_port=8080,
        threshold=0.85,
        position=-1,
    ):
        super().__init__(kind=OpKind.JOIN, position=position)
        self.right_table = right_table
        self.service_address = service_address
        self.service_port = service_port
        self.threshold = threshold
        self._indexed_rows = [
            (idx, right.input.data)
            for idx, right in enumerate(self.right_table)
        ]
        self.cp_id = f"icp:{position}:{id(self)}"

    def _post_icp(self, path, payload):
        return requests.post(
            f"http://{self.service_address}:{self.service_port}/{path}",
            json=payload,
            timeout=60,
        )

    def _build_remote_index(self) -> None:
        response = self._post_icp(
            "build_index",
            {
                "cp_id": self.cp_id,
                "right_table": [list(row) for row in self._indexed_rows],
            },
        )
        response.raise_for_status()

    def _query_retrieval(self, left_row: tuple[Any, ...]) -> list[dict[str, Any]]:
        payload = {
            "cp_id": self.cp_id,
            "left_tuple": list(left_row),
            "threshold": self.threshold,
        }

        response = self._post_icp("query", payload)
        if response.ok:
            return response.json()["results"]
        
        self._build_remote_index()

        retry_response = self._post_icp("query", payload)
        retry_response.raise_for_status()
        return retry_response.json()["results"]

    def __call__(self, ctx):
        if not self.right_table:
            return []

        left_row = (ctx.state.idx, ctx.input.data)
        try:
            ranked_rows = self._query_retrieval(left_row)
        except requests.RequestException:
            ranked_rows = [
                {"metadata": [idx], "score": 0.0}
                for idx in range(len(self.right_table))
            ]

        out = []
        for row in ranked_rows:
            metadata = row["metadata"]
            right_idx = int(metadata[0])
            right = self.right_table[right_idx]
            base_input = SemanticInput(
                data=ctx.input.data[:],
                right_data=[],
            )
            _input = base_input.add_right(right.input.data)
            new_ctx = SemContext(
                input=_input,
                state=ExecutionState(
                    raw_request=ctx.state.raw_request,
                    pin_req_id=None,
                    executor=ctx.state.executor,
                ),
            )
            out.append(new_ctx)

        return out

        
class SemClassify(BaseOp):
    LOG = False

    def __init__(self, classes, pin=False, unpin=False, position=-1):
        super().__init__(kind=OpKind.TUPLE_INDEPENDENT, position=position)
        self.classes = list(classes)
        self.pin = pin
        self.unpin = unpin
        self.max_tokens = max(KVMemoryManager.get_instance().token_length(g) for g in self.classes) + 1

        self.instruction = "\n\n" \
                + "Choose exactly one class from the list below.\n" \
                + f"Class: {', '.join(self.classes)}\n" \
                + "Answer with the class name only:"
        self.instruction_token_len = KVMemoryManager.get_instance().token_length(self.instruction) + self.max_tokens + 1


    def _build_prompts(self, ctx):
        data_prompt = ctx.input.data if 'system' in ctx.input.data[0]['role'] else get_system_prompt() + ctx.input.data
        full_prompt = get_prompt(self.instruction, ctx.input.data, op=OpName.SEM_FILTER)
        return data_prompt, full_prompt


    def estimate_tokens(self, ctx):
        _, prompt = self._build_prompts(ctx)
            
        prompt_str =KVMemoryManager.get_instance().apply_chat_template(prompt)
        prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

        return prompt_token_len + self.max_tokens


    async def __call__(self, ctx: SemContext, priority: int = 0) -> SemContext: 
 
        data_part, prompt = self._build_prompts(ctx)

        data_result = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=data_part,
                max_tokens=1,
                pin=self.pin,
                priority=priority,
        )

        output = await ctx.state.executor.execute(
                raw_request=ctx.state.raw_request,
                prompt=prompt,
                max_tokens=self.max_tokens,
                pin=False,
                priority=priority,
        )
        group_output = output.text.strip().lower()
        group = ""
        for g in self.classes:
            if g.lower() in group_output:
                group = g
                break 

        ctx.input.data = prompt[:-1]
        ctx.output.append({
            str(self.__class__): str(group)
        })

        if self.pin:
            ctx.state.pin_req_id = data_result.request_id

        elif self.unpin and ctx.state.pin_req_id is not None:
            if self.LOG:
                print(
                    "[sem-op] "
                    f"SemClassify unpin pin_req_id={ctx.state.pin_req_id}"
                )
            await ctx.state.executor.unpin(ctx.state.raw_request, ctx.state.pin_req_id)
            ctx.state.pin_req_id = None

        return ctx


class SemAgg(BaseOp):
    def __init__(self, instruction: str, max_tokens: int = 8192, concurrency: int = 8, position=-1):
        super().__init__(kind=OpKind.BLOCKING, position=position)
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.concurrency = concurrency

    def _ctx_to_text(self, ctx: SemContext) -> str:
        parts = []
        for message in ctx.input.data:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)

    def _build_prompt(self, chunk: List[SemContext]):
        docs = []
        for i, ctx in enumerate(chunk, 1):
            docs.append(f"Document {i}:\n{self._ctx_to_text(ctx)}")
        data = [{
            "role": "user",
            "type": "text",
            "content": "\n\n".join(docs),
        }]
        return get_prompt(self.instruction, data, op=OpName.SEM_AGG)


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
        context_chunks = []
        current_chunk = []
        instruction_overhead = KVMemoryManager.get_instance().token_length(
            "\n\n" + self.instruction + "\n\n"
        )
        per_doc_overhead = KVMemoryManager.get_instance().token_length(
            "\n\nDocument 1:\n"
        )
        current_chunk_tokens = instruction_overhead

        for ctx in ctxs:
            ctx_tokens = KVMemoryManager.get_instance().token_length(ctx.input.data)
            chunk_item_tokens = ctx_tokens + per_doc_overhead
            # Allow an oversized single item as its own chunk instead of failing early.
            if current_chunk and current_chunk_tokens + chunk_item_tokens > self.max_tokens:
                context_chunks.append(current_chunk)
                current_chunk = []
                current_chunk_tokens = instruction_overhead

            current_chunk.append(ctx)
            current_chunk_tokens += chunk_item_tokens

        if current_chunk:
            context_chunks.append(current_chunk)

        return context_chunks


    def _build_reducer(self, chunk: List[SemContext]):
        parent = self
        
        class Reducer:
            def __init__(self, chunk: List[SemContext]):
                self.chunk = chunk

                prompt = parent._build_prompt(chunk)
                prompt_str = KVMemoryManager.get_instance().apply_chat_template(prompt)
                prompt_token_len = KVMemoryManager.get_instance().token_length(prompt_str)

                self.budget = (prompt_token_len + parent.max_tokens) * KVMemoryManager.get_instance().bytes_per_token 

            async def __call__(self) -> SemContext:
                return await parent._reduce_chunk(self.chunk)

        return Reducer(chunk)


    async def _reduce_chunk(self, chunk: List[SemContext]) -> SemContext:
        executor = chunk[0].state.executor
        raw_request = chunk[0].state.raw_request

        prompt = self._build_prompt(chunk)

        result = await executor.execute(
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
    def __init__(self, instruction: str, k: int= 10, concurrency: int = 20, position=-1):
        super().__init__(kind=OpKind.BLOCKING, position=position)
        
        self.instruction = instruction
        self.k = k
        self.max_tokens = 5
        self.concurrency = concurrency

    def _ctx_to_text(self, ctx: SemContext) -> str:
        parts = []
        for message in ctx.input.data:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)

    def _build_compare_prompt(self, first: SemContext, second: SemContext):
        data = [{
            "role": "user",
            "type": "text",
            "content": (
                f"Document A:\n{self._ctx_to_text(first)}\n\n"
                f"Document B:\n{self._ctx_to_text(second)}"
            ),
        }]
        return get_prompt(self.instruction, data, op=OpName.SEM_TOPK)

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
        parent = self

        def build_task(other: SemContext):
            raw_request = pivot.state.raw_request
            max_tokens = self.max_tokens

            class CompareTask:
                def __init__(self):
                    self.prompt = parent._build_compare_prompt(pivot, other)
                    tokens = KVMemoryManager.get_instance().token_length(self.prompt)
                    self.budget = tokens * KVMemoryManager.get_instance().bytes_per_token


                async def __call__(self):
                    result = await executor.execute(
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
        prompt = self._build_compare_prompt(a, b)

        result = await executor.execute(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=self.max_tokens,
            pin=False,
        )
        return result.text.strip().upper().startswith("A")
