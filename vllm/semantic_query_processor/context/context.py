
from typing import Any, Dict
from dataclasses import dataclass


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
