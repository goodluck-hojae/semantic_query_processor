import json

from click import prompt
from vllm import SamplingParams

# TODO: Remove the dependency 
from vllm.semantic_query_processor.query import Query
from vllm.entrypoints.openai.protocol import CompletionRequest
from vllm.entrypoints.openai.api_server import create_completion
from fastapi import Request

from dataclasses import dataclass, field
from typing import Any, Dict


OPERATOR_LIST = ['sem_filter', 'sem_join', 'sem_groupby', 'sem_topk', 'sem_map', 'sem_agg']
UNCETAIN_OPERATOR = ['sem_filter', 'sem_join']


@dataclass(frozen=True)
class SemanticInput:
    data: str
    token_len: int
 

@dataclass
class ExecutionState:
    raw_request: Any
    pin_req_id: Any


@dataclass
class SemContext:
    input: SemanticInput
    state: ExecutionState


def build_completion_request(prompt, max_tokens, pin=False):
    return CompletionRequest(
        # model="meta-llama/Llama-3.1-8B-Instruct",
        model="meta-llama/Llama-3.2-3B-Instruct",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=False,
        vllm_xargs={
            "pin_kv": pin,
        }
    )

async def completion_call_internal(raw_request: Request, prompt, max_tokens, pin=False):
    req = build_completion_request(prompt, max_tokens, pin=pin)

    gen = await create_completion(
        request=req,
        raw_request=raw_request
    )
    
    raw = gen.body.decode("utf-8")
    data = json.loads(raw)
    return data, req
    
    
async def evict_request(raw_request, req_id: str):
    print('evicting request:', req_id)
    async_llm = raw_request.app.state.engine_client  # AsyncLLM
    await async_llm.abort(req_id)

 

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
        for op in self.ops:
            if next:    
                next = await op(self.ctx)




class BaseOp:
    max_len: int
    is_last: bool = False

    async def __call__(self, ctx):
        raise NotImplementedError


class SemFilter(BaseOp):
    def __init__(self, instruction, pin=False, is_last=False, max_len=10):   
        self.instruction = instruction
        self.pin = pin
        self.is_last = is_last
        self.max_len = max_len

    async def __call__(self, ctx):
        prompt = (
            ctx.input.data
            + "\n\n"
            + self.instruction
            + "\n\n"
            + "Answer yes or no. Answer:"
        )

        res, req = await completion_call_internal(
            ctx.state.raw_request,
            prompt,
            self.max_len,
            pin=(self.pin and not self.is_last)
        )
        ctx.state.pin_req_id = res["id"]

        # if the answer is no, we can evict the pinned kv immediately
        if False:
            engine = ctx.raw_request.app.state.engine_client
            await engine.engine_core.call_utility_async(
                "unpin_kv",
                ctx.state.pin_req_id,
            )

        return True

    
class SemMap(BaseOp):
    def __init__(self, instruction, pin=False, is_last=False, max_len=128):   
        self.instruction = instruction
        self.pin = pin
        self.is_last = is_last
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
            pin=(self.pin and not self.is_last)
        )

        # sem_map always unpins the previous pin_req_id
        if ctx.state.pin_req_id is not None:
            engine = ctx.state.raw_request.app.state.engine_client
            await engine.engine_core.call_utility_async(
                "unpin_kv",
                ctx.state.pin_req_id,
            )

        # TODO impl pin gen request if needed
        pass
        return res["choices"][0]["text"]
