from pathlib import Path

from vllm.semantic_query_processor.budget import KVMemoryManager
from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.controller import SemanticPlan
from vllm.semantic_query_processor.execution.vllm_executor import VLLMExecutor


class QueryProcessor:
    def __init__(self, model_name, budget):
        self.model_name = model_name
        KVMemoryManager.init(model_name, budget)
        self.executor = VLLMExecutor(model=model_name)


    def parse(self, query: Query):
        operations = [] # An operation consists of (data, operator) pairs 
        return operations


    # TODO: Organize operations into a plan
    def plan(self, query: Query):
        print(f"[QueryProcessor] Planning for query: {query.query}")
        return query


    def _data_source(self, raw_request, query: Query):
        path = Path(query.data_path)
 
        if path.suffix.lower() == ".csv":
            for ctx in self._csv_reader(raw_request, path):
                yield ctx

 
    async def execute(self, raw_request, query: Query):
        

        plan = SemanticPlan(self.executor)
        await plan.execute(raw_request, query)

        return "ctxs"
