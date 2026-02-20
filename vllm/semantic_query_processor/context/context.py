
from typing import Any, Dict, List
from dataclasses import dataclass, field
from vllm.semantic_query_processor.execution.vllm_executor import LLMExecutor


@dataclass(frozen=True)
class SemanticInput:
    data: Any = None
    token_len: int = -1
    left_input: str = None
    right_input: str = None
 

@dataclass
class ExecutionState:
    raw_request: Any = None
    pin_req_id: Any = None
    executor: LLMExecutor = None
    predicate: bool = True
    group: str = ""
    idx: int=-1
    

@dataclass
class SemContext:
    input: SemanticInput = None
    output: List[Dict[str, Any]] = field(default_factory=list)
    state: ExecutionState = None
