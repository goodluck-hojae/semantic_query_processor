from pathlib import Path
import csv

from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.data import data
from vllm.semantic_query_processor.pipeline import SemanticChain
from vllm.semantic_query_processor.context import SemContext, SemanticInput, ExecutionState


def _data_source(raw_request, query: Query, kv_estimator):
    path = Path(query.data_path)

    if path.suffix.lower() == ".csv":
        for ctx in _csv_reader(raw_request, path, kv_estimator):
            yield ctx


def _csv_reader(raw_request, path: Path, kv_estimator):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[:100]
        for row in rows:
            yield SemContext(
                input=SemanticInput(
                        data=str(row['Resume_str']).strip(),
                        token_len=kv_estimator.token_length(str(row['Resume_str']).strip()),
                    ),
                state=ExecutionState(
                    raw_request=raw_request,
                    pin_req_id=None,
                )
            )
