from itertools import chain
from pathlib import Path
import csv
from typing import Any

from pyparsing import Optional

from vllm.semantic_query_processor.sem_ops import ops
from vllm.semantic_query_processor.cost_estimator import KVConfig, KVEstimator
from vllm.semantic_query_processor.query import Query

import asyncio


class QueryProcessor:
    def __init__(self, model_name, budget):
        self.model_name = model_name
        self.kv_estimator = KVEstimator(model_name, budget)

    # When engine is initialized, it acquires a sample request to access engine core
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
                yield ops.SemContext(
                    raw_request=raw_request,
                    data=str(row['Resume_str']).strip(),
                    token_length=self.kv_estimator.token_length(str(row['Resume_str']).strip()),
                    question={
                        "sem_filter": "Is the candidate capable of GPU programming?",
                        "sem_map": "Summarize the following resume.",
                    },
                    prefix_req_id=None,
                    prefix=True
            )


    def _build_chain(self, ctx) -> ops.SemanticChain:
        return ops.SemanticChain(
            ctx,
            ops.SemFilter(),
            ops.SemMap(),
            bytes_per_token=self.kv_estimator.bytes_per_token,
        )
    

    async def execute(self, raw_request, query: Query):
        ctx_iter = self._data_source(raw_request, query)

        await self._run_workers(ctx_iter)


    async def _run_workers(self, ctx_iter, concurrency=100):

        queue = asyncio.Queue(maxsize=concurrency)
        capacity_cond = asyncio.Condition()

        async def worker():
            while True:
                chain = await queue.get()
                try:
                    await chain()
                finally:
                    async with capacity_cond:
                        self.kv_estimator.release(chain.budget)
                        capacity_cond.notify_all()
                    queue.task_done()

        # start workers
        workers = [
            asyncio.create_task(worker())
            for _ in range(concurrency)
        ]

        for ctx in ctx_iter:
            chain = self._build_chain(ctx)

            # WAIT until capacity available
            async with capacity_cond:
                while not self.kv_estimator.can_admit(chain.budget):
                    await capacity_cond.wait()

                self.kv_estimator.allocate(chain.budget)

            await queue.put((chain))

        # wait until all tasks processed
        await queue.join()

        # shutdown workers
        for w in workers:
            w.cancel()
























    async def execute_ref(self, raw_request, query: Query):
        path = Path(query.data_path)
        if not path.exists():
            raise FileNotFoundError(path)

        print(f"[QueryProcessor] Query: {query.query}")
        print(f"[QueryProcessor] Scanning: {query.data_path}")

        # CSV case
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                batch = []

                #TODO: window approach
                rows = list(reader)[:200]
                
                chain = ops.sem_chain(ops.sem_filter)
                for row in rows:
                    data = str(row['Resume_str']).strip()

                    ctx = ops.SemContext(
                        raw_request=raw_request,
                        data=data,
                        token_length=self.kv_estimator.token_length(str(row['Resume_str']).strip()),
                        question={
                            "sem_filter": "Is the candidate capable of GPU programming?",
                            "sem_map": "Sumamrize the following resume.",
                        },
                        prefix_req_id=None,
                        prefix=False
                    )
                    batch.append(ctx)

                tasks = [chain(ctx) for ctx in batch]
                results = await asyncio.gather(*tasks)
                batch.clear()

                print('filter done')


                #TODO: window approach
                chain = ops.sem_chain(ops.sem_map)
                for row in rows:
                    data = str(row['Resume_str']).strip()

                    ctx = ops.SemContext(
                        raw_request=raw_request,
                        data=data,
                        token_length=self.kv_estimator.token_length(str(row['Resume_str']).strip()),
                        question={
                            "sem_filter": "Is the candidate capable of GPU programming?",
                            "sem_map": "Sumamrize the following resume.",
                        },
                        prefix_req_id=None,
                        prefix=False
                    )
                    batch.append(ctx)

                tasks = [chain(ctx) for ctx in batch]
                results = await asyncio.gather(*tasks)
                batch.clear()

