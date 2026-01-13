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


@dataclass
class SemContext:
    raw_request: Any
    data: Any
    token_length: Any
    question: Any
    prefix_req_id: Any
    prefix: Any


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


OPERATOR_LIST = ['sem_filter', 'sem_join', 'sem_groupby', 'sem_topk', 'sem_map', 'sem_agg']
UNCETAIN_OPERATOR = ['sem_filter', 'sem_join']



class SemanticChain:
    def __init__(self, ctx, *ops, bytes_per_token: int):
        self.ctx = ctx
        self.ops = ops
        self.bytes_per_token = bytes_per_token
        self.budget = self.estimate_token_budget(ctx.token_length)
        

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
        value = self.ctx

        for op in self.ops:
            reuse = await op(value)

            if isinstance(reuse, tuple):
                value.prefix_req_id = reuse[1]
                reuse = reuse[0]

            if not reuse:
                break

        return value



class BaseOp:
    max_len: int

    async def __call__(self, ctx):
        raise NotImplementedError


class SemFilter(BaseOp):
    max_len = 10

    async def __call__(self, ctx):
        prompt = (
            ctx.data
            + "\n\n"
            + ctx.question["sem_filter"]
            + "\n\n"
            + "Answer yes or no. Answer:"
        )

        res, req = await completion_call_internal(
            ctx.raw_request,
            prompt,
            self.max_len,
            pin=ctx.prefix
        )

        return (True, res["id"])

    
class SemMap(BaseOp):
    max_len = 128

    async def __call__(self, ctx):
        prompt = (
            ctx.data
            + "\n\n"
            + ctx.question["sem_map"]
            + "\n\n"
        )

        res, req = await completion_call_internal(
            ctx.raw_request,
            prompt,
            self.max_len,
            pin=False
        )

        if ctx.prefix:
            engine = ctx.raw_request.app.state.engine_client
            await engine.engine_core.call_utility_async(
                "unpin_kv",
                ctx.prefix_req_id
            )

        return res["choices"][0]["text"]
