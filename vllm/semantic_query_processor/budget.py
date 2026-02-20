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
        self._lock = asyncio.Lock()

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


    async def can_admit(self, budget: int) -> bool:
        async with self._lock:
            return budget <= self._capacity

    async def allocate(self, budget: int):
        async with self._lock:
            if budget > self._capacity:
                return False
            self._capacity -= budget
            return True

    async def release(self, budget: int):
        async with self._lock:
            self._capacity += budget
            return True

    async def execute_tasks(self, seeds, task_builder, concurrency=20):
        queue = asyncio.Queue(maxsize=concurrency)
        capacity_cond = asyncio.Condition()
        results = []

        async def worker():
            while True:
                task = await queue.get()
                try:
                    await task()
                    if task.ctx.state.predicate:
                        results.append(task.ctx)
                finally:
                    async with capacity_cond:
                        await self.release(task.budget)
                        capacity_cond.notify_all() 
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]

        for seed in seeds:
            task = task_builder(seed)

            async with capacity_cond:
                while not await self.can_admit(task.budget):
                    await capacity_cond.wait()
                await self.allocate(task.budget)

            await queue.put(task)

        await queue.join()

        for w in workers:
            w.cancel()

        return results
