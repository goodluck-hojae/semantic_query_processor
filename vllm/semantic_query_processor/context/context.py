
from typing import Any, Dict
from dataclasses import dataclass
from vllm.semantic_query_processor.execution.vllm_executor import LLMExecutor


@dataclass(frozen=True)
class SemanticInput:
    data: str
    token_len: int
 

@dataclass
class ExecutionState:
    raw_request: Any
    pin_req_id: Any
    executor: LLMExecutor


@dataclass
class SemContext:
    input: SemanticInput
    state: ExecutionState
