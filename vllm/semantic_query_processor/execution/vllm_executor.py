# execution/vllm_executor.py
import json
from fastapi import Request
from vllm.entrypoints.openai.protocol import CompletionRequest
from vllm.entrypoints.openai.api_server import create_completion

from .executor import LLMExecutor, CompletionResult


class VLLMExecutor(LLMExecutor):

    UNPIN_FUNCTION = 'unpin_request'
    
    def __init__(self, model: str):
        self.model = model

    def _build_request(self, prompt, max_tokens, pin):
        return CompletionRequest(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            vllm_xargs={"pinned": pin},
        )

    async def complete(
        self,
        *,
        raw_request: Request,
        prompt: str,
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
