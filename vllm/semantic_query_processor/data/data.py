from pathlib import Path
import csv
import json
from pathlib import Path
from vllm.semantic_query_processor.query import Query
from vllm.semantic_query_processor.data import data
from vllm.semantic_query_processor.context import SemContext, SemanticInput, ExecutionState
from vllm.semantic_query_processor.budget import KVMemoryManager


# def _data_source(raw_request, query: Query, executor):
#     path = Path(query.data_path)

#     if path.suffix.lower() == ".csv":
#         for ctx in _message_reader(raw_request, path, executor):
#             yield ctx

def _data_source(raw_request, data_path, executor):
    path = Path(data_path)

    if path.is_dir():
        for txt_path in sorted(path.glob("*.txt"), key=lambda p: int(p.stem)):
            yield from _message_reader(raw_request, txt_path, executor)

    elif path.suffix.lower() == ".json":
        yield from _message_reader(raw_request, path, executor)

    elif path.suffix.lower() == ".csv":
        for ctx in _message_reader(raw_request, path, executor):
            yield ctx


def _message_reader(raw_request, path: Path, executor):
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    yield SemContext(
        input=SemanticInput(
            data=text,
            token_len=-1,
        ),
        state=ExecutionState(
            raw_request=raw_request,
            pin_req_id=None,
            executor=executor,
            idx=int(path.stem),
        ),
    )

def research_category_data():
    
    categories = ['ai', 'biology', 'chemistry', 'geology', 'math', 'phyics']
    out = []
    for category in categories:
        
        ctx = SemContext(
            input=SemanticInput(
                data=category,
                token_len=-1,
            ),
            state=ExecutionState(
                raw_request=None,
                pin_req_id=None,
                executor=None,
                idx=-1,
            )
        )
        out.append(ctx)
    return out
                
# def _message_reader(raw_request, path: Path, executor):
#     with path.open("r", encoding="utf-8") as f:
#         messages = json.load(f)

#     prompt = KVMemoryManager.get_instance().tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=False,
#     )
#     # messages must be a list[dict] like [{"role":..., "type":..., "content":...}, ...]
#     yield SemContext(
#         input=SemanticInput(
#             data=messages,
#             token_len=KVMemoryManager.get_instance().token_length(prompt),
#         ),
#         state=ExecutionState(
#             raw_request=raw_request,
#             pin_req_id=None,
#             executor=executor,
#             idx=int(path.name.split('.')[0])
#         ),
#     )



def _csv_reader(raw_request, path: Path, executor):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[:20]
        for row in rows:
            yield SemContext(
                input=SemanticInput(
                        data=str(row['Resume_str']).strip(),
                        token_len=KVMemoryManager.get_instance().token_length(str(row['Resume_str']).strip()),
                    ),
                state=ExecutionState(
                    raw_request=raw_request,
                    pin_req_id=None,
                    executor =executor
                )
            )
