# execution/vllm_executor.py
import json
from fastapi import Request
from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest
from vllm.entrypoints.openai.api_server import create_completion, create_chat_completion

from .executor import LLMExecutor, CompletionResult


class VLLMExecutor(LLMExecutor):

    UNPIN_FUNCTION = 'unpin_request'
    
    def __init__(self, model: str):
        self.model = model

    def _build_chat_request(self, message, max_tokens, pin):
        return ChatCompletionRequest(
            model=self.model,
            messages=message,
            max_tokens=max_tokens,
            temperature=0.0,
            vllm_xargs={"pinned": pin},
        )


    def _build_request(self, prompt, max_tokens, pin):
        return CompletionRequest(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            vllm_xargs={"pinned": pin},
        )

    async def execute(
        self,
        raw_request: Request,
        prompt,
        max_tokens: int,
        pin: bool = False,
    ) -> CompletionResult:
        llm_func = self.complete if type(prompt) is str else self.chatcomplete
        return await llm_func(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=max_tokens,
            pin=pin,
        )


    async def chatcomplete(
        self,
        raw_request: Request,
        prompt,
        max_tokens: int,
        pin: bool = False,
    ) -> CompletionResult:
        req = self._build_chat_request(prompt, max_tokens, pin)

        gen = await create_chat_completion(
            request=req,
            raw_request=raw_request,
        )

        raw = gen.body.decode("utf-8")
        data = json.loads(raw)

        return CompletionResult(
            text=data['choices'][0]['message']['content'],
            request_id=data["id"],
            finish_reason=data["choices"][0].get("finish_reason", "")
        )


    async def complete(
        self,
        *,
        raw_request: Request,
        prompt,
        max_tokens: int,
        pin: bool = False,
    ) -> CompletionResult:
        req = self._build_request(prompt, max_tokens, pin)

        gen = await create_completion(
            request=req,
            raw_request=raw_request,
        )

        raw = gen.body.decode("utf-8")
        data = json.loads(raw)

        return CompletionResult(
            text=data["choices"][0]["text"],
            request_id=data["id"],
            finish_reason=data["choices"][0].get("finish_reason", ""),
        )

    async def unpin(self, raw_request: Request, request_id: str):
        engine = raw_request.app.state.engine_client
        await engine.engine_core.call_utility_async(
            VLLMExecutor.UNPIN_FUNCTION,
            request_id,
        )

    async def abort(self, raw_request: Request, request_id: str):
        engine = raw_request.app.state.engine_client
        await engine.abort(request_id)
