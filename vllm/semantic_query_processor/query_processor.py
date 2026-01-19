from pathlib import Path
import csv

from vllm.semantic_query_processor.sem_ops import ops
from vllm.semantic_query_processor.cost_estimator import KVEstimator
from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.context import SemContext, SemanticInput, ExecutionState
from vllm.semantic_query_processor.pipeline import SemanticPipeline


class QueryProcessor:
    def __init__(self, model_name, budget):
        self.model_name = model_name
        self.kv_estimator = KVEstimator(model_name, budget)


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


    def _csv_reader(self, raw_request, path: Path):
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)[:200]
            for row in rows:
                yield SemContext(
                    input=SemanticInput(
                            data=str(row['Resume_str']).strip(),
                            token_len=self.kv_estimator.token_length(str(row['Resume_str']).strip()),
                        ),
                    state=ExecutionState(
                        raw_request=raw_request,
                        pin_req_id=None,
                    )
                )


    async def execute(self, raw_request, query: Query):
        ctxs = list(self._data_source(raw_request, query))

        operators = (
            ops.SemFilter("Is the candidate capable of GPU programming?", pin=True),
            ops.SemMap("Summarize the following resume.", is_last=True),
        )
        pipeline = SemanticPipeline(self.kv_estimator)
        await pipeline.execute(raw_request, query)

        return ctxs
