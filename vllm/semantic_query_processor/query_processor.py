from pathlib import Path
import csv
from typing import Any

from pyparsing import Optional

from vllm.semantic_query_processor.sem_ops import ops
from vllm.semantic_query_processor.cost_estimator import KVConfig, KVEstimator
from vllm.semantic_query_processor.query import Query

import asyncio


class QueryProcessor:
    def __init__(self):
        
        self.kv_estimator = KVEstimator(
            kv_config=KVConfig(
                    bytes_per_token=16,
                    max_classifiy_tokens=10,
                    max_gen_tokens=8192*2),
            max_kv_bytes=2 * 1024 * 1024 # * 1024,  # 8 GB
        )


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
                    question={
                        "sem_filter": "Is the candidate capable of GPU programming?",
                        "sem_map": "Summarize the following resume.",
                    },
                    prefix_req_id=None,
                    prefix=True
            )


    def _build_chain(self, query: Query):
        # operators = self.parse(query)
        return ops.sem_chain(
            ops.sem_filter,
            ops.sem_map,
        )
    

    async def _run_workers(self, chain, ctx_iter, concurrency=40):

        queue = asyncio.Queue(maxsize=concurrency)

        async def worker():
            while True:
                ctx = await queue.get()
                try:
                    await chain(ctx)
                finally:
                    queue.task_done()

        # start workers
        workers = [
            asyncio.create_task(worker())
            for _ in range(concurrency)
        ]

        # PRODUCE (SYNC ITERATION)
        for ctx in ctx_iter:          # <-- normal for-loop
            await queue.put(ctx)

        # wait until all tasks processed
        await queue.join()

        # shutdown workers
        for w in workers:
            w.cancel()


    async def execute(self, raw_request, query: Query):
        chain = self._build_chain(query)
        ctx_iter = self._data_source(raw_request, query)

        await self._run_workers(chain, ctx_iter)


        # path = Path(query.data_path)
        # if not path.exists():
        #     raise FileNotFoundError(path)

        # print(f"[QueryProcessor] Query: {query.query}")
        # print(f"[QueryProcessor] Scanning: {query.data_path}")

        # MAX_CONCURRENCY = 40
        # queue = asyncio.Queue(maxsize=MAX_CONCURRENCY)
        # chain = ops.sem_chain(ops.sem_filter, ops.sem_map)

        
        # async def worker(worker_id: int):
        #     while True:
        #         ctx = await queue.get()
        #         try:
        #             await chain(ctx)
        #         finally:
        #             queue.task_done()

        # # start workers
        # workers = [
        #     asyncio.create_task(worker(i))
        #     for i in range(MAX_CONCURRENCY)
        # ]



        # if path.suffix.lower() == ".csv":
        #     with path.open("r", encoding="utf-8", newline="") as f:
        #         reader = csv.DictReader(f)
        #         batch = []

        #         #TODO: window approach
        #         rows = list(reader)
        #         chain = ops.sem_chain(ops.sem_filter, ops.sem_map)
        #         for row in rows:
        #             data = str(row['Resume_str']).strip()

        #             ctx = ops.SemContext(
        #                 raw_request=raw_request,
        #                 data=data,
        #                 question={
        #                     "sem_filter": "Is the candidate capable of GPU programming?",
        #                     "sem_map": "Sumamrize the following resume.",
        #                 },
        #                 prefix_req_id=None,
        #                 prefix=True
        #             )

        #             await queue.put(ctx)

        # await queue.join()

        # # shutdown workers
        # for w in workers:
        #     w.cancel()



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

