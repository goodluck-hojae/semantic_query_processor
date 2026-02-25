import csv
from pathlib import Path
import threading
import asyncio
import torch
from transformers import AutoConfig, AutoTokenizer


def compute_bytes_per_token(
    model_name: str,
    dtype: torch.dtype = torch.float16,
) -> int:
    cfg = AutoConfig.from_pretrained(model_name)

    num_layers = cfg.num_hidden_layers

    # GQA / MQA aware
    num_kv_heads = getattr(
        cfg,
        "num_key_value_heads",
        cfg.num_attention_heads,
    )

    head_dim = cfg.hidden_size // cfg.num_attention_heads

    dtype_bytes = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
    }[dtype]

    return (
        2 *                 
        num_layers *
        num_kv_heads *
        head_dim *
        dtype_bytes
    )

class KVMemoryManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_name, kv_capacity, dtype):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bytes_per_token = compute_bytes_per_token(model_name, dtype)
        self._capacity = kv_capacity * 0.95
        self._stage_capacity = {}
        self._stage_used = {}
        self._lock = asyncio.Lock()


    def register_stage(self, stage_id: str, fraction: float):
        cap = self._capacity * fraction
        self._stage_capacity[stage_id] = cap
        self._stage_used[stage_id] = 0
        
    @classmethod
    def init(cls, model_name, kv_capacity, dtype=torch.float16):
        with cls._lock:
            if cls._instance is not None:
                raise RuntimeError("KVMemoryManager already initialized")
            cls._instance = cls(model_name, kv_capacity, dtype)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("KVMemoryManager not initialized")
        return cls._instance


    def token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


    async def can_admit(self, stage_id, budget):
        async with self._lock:
            return (
                self._stage_used[stage_id] + budget
                <= self._stage_capacity[stage_id]
            )

    async def allocate(self, stage_id, budget):
        async with self._lock:
            if self._stage_used[stage_id] + budget > self._stage_capacity[stage_id]:
                return False

            self._stage_used[stage_id] += budget
            return True


    async def release(self, stage_id, budget):
        async with self._lock:
            self._stage_used[stage_id] -= budget

    # async def can_admit(self, budget: int) -> bool:
    #     async with self._lock:
    #         return budget <= self._capacity

    # async def allocate(self, budget: int):
    #     async with self._lock:
    #         if budget > self._capacity:
    #             return False
    #         self._capacity -= budget
    #         return True

    # async def release(self, budget: int):
    #     async with self._lock:
    #         self._capacity += budget
    #         return True
    
    async def execute_tasks(self, seeds, task_builder, concurrency=20):

        queue = asyncio.Queue(maxsize=concurrency)
        capacity_cond = asyncio.Condition()
        final_results = []

        async def worker():
            while True:
                pipeline = await queue.get()
                try:
                    out = await pipeline()

                    # CASE 1: fan-out
                    if isinstance(out, list):
                        for child in out:

                            async with capacity_cond:
                                while not await self.can_admit(
                                    child.stage_id, child.budget
                                ):
                                    await capacity_cond.wait()

                                await self.allocate(
                                    child.stage_id, child.budget
                                )

                            await queue.put(child)

                    # CASE 2: leaf
                    else:
                        final_results.append(pipeline.ctx)

                finally:
                    async with capacity_cond:
                        await self.release(
                            pipeline.stage_id, pipeline.budget
                        )
                        capacity_cond.notify_all()

                    queue.task_done()

        workers = [
            asyncio.create_task(worker())
            for _ in range(concurrency)
        ]

        # -----------------------------
        # Submit tasks with stage quota
        # -----------------------------
        for seed in seeds:

            task = task_builder(seed)

            async with capacity_cond:
                while not await self.can_admit(task.stage_id, task.budget):
                    await capacity_cond.wait()

                await self.allocate(task.stage_id, task.budget)

            await queue.put(task)

        await queue.join()

        for w in workers:
            w.cancel()

        return final_results
