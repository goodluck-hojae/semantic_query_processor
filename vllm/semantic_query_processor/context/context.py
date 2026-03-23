
from typing import Any, Dict, List
from dataclasses import dataclass, field, replace
from vllm.semantic_query_processor.execution.vllm_executor import LLMExecutor
from vllm.semantic_query_processor.sem_ops.prompt_utils import get_data_prompt

 
class SemanticInput:
    def __init__(self, data=None, token_len=-1, right_data=[]):
        if type(data) is str:
            data = get_data_prompt(data)
        self.data = data
        self.token_len = token_len
        self.right_data = []
        for i in range(len(right_data)):
            self.right_data += right_data[i]

    def add_right(self, value):
        self.right_data += value
        return self
    
    
@dataclass
class ExecutionState:
    raw_request: Any = None
    pin_req_id: Any = None
    executor: LLMExecutor = None
    predicate: bool = True
    idx: int=-1
    

@dataclass
class SemContext:
    input: SemanticInput = None
    output: List[Dict[str, Any]] = field(default_factory=list)
    state: ExecutionState = None
