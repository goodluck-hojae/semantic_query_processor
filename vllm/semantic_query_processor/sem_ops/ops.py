import json
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


def sem_chain(*ops):
    async def run(x):
        value = x
        for op in ops:
            reuse = await op(value)
            if type(reuse) == tuple:
                value.prefix_req_id = reuse[1]
            if reuse:
                continue
            else:
                break
    return run


async def sem_filter(ctx: SemContext):
    prompt = ctx.data + '\n\n' + ctx.question['sem_filter'] + '\n\n' + 'Answer yes or no. Answer:'
    res, req = await completion_call_internal(ctx.raw_request, prompt, 10, pin=ctx.prefix)
    res['choices'][0]['text']
    return (True, res['id'])
    

async def sem_map(ctx: SemContext):
    prompt = ctx.data + '\n\n' + ctx.question['sem_map'] + '\n\n'
    res, req = await completion_call_internal(ctx.raw_request, prompt, 128, pin=False)
    if ctx.prefix:
        await ctx.raw_request.app.state.engine_client.engine_core.call_utility_async('unpin_kv', ctx.prefix_req_id)
    return res['choices'][0]['text']

