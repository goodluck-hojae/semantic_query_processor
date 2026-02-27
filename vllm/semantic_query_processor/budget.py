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
    _init_lock = threading.Lock()

    def __init__(self, model_name, kv_capacity, dtype):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bytes_per_token = compute_bytes_per_token(model_name, dtype)

        self._capacity = kv_capacity * 0.95
        self._global_used = 0

        self._stage_capacity = {}
        self._stage_used = {}

        self._cond = asyncio.Condition()

    # Stage API (Pipeline)
    def register_stage(self, stage_id: int, fraction: float):
        self._stage_capacity[stage_id] = self._capacity * fraction
        self._stage_used[stage_id] = 0

    async def can_admit_stage(self, stage_id: int, budget: int):
        async with self._cond:
            return (
                self._stage_used[stage_id] + budget
                <= self._stage_capacity[stage_id]
            )

    async def allocate_stage(self, stage_id: int, budget: int):
        async with self._cond:
            while (
                self._stage_used[stage_id] + budget
                > self._stage_capacity[stage_id]
            ):
                await self._cond.wait()

            self._stage_used[stage_id] += budget

    async def release_stage(self, stage_id: int, budget: int):
        async with self._cond:
            self._stage_used[stage_id] -= budget
            if self._stage_used[stage_id] < 0:
                self._stage_used[stage_id] = 0

            self._cond.notify_all()

    # Global API (Blocking)
    async def can_admit(self, budget: int):
        async with self._cond:
            return self._global_used + budget <= self._capacity

    async def allocate(self, budget: int):
        async with self._cond:
            while self._global_used + budget > self._capacity:
                await self._cond.wait()

            self._global_used += budget

    async def release(self, budget: int):
        async with self._cond:
            self._global_used -= budget
            if self._global_used < 0:
                self._global_used = 0

            self._cond.notify_all()


    def token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @classmethod
    def init(cls, model_name, kv_capacity, dtype=torch.float16):
        with cls._init_lock:
            if cls._instance is not None:
                raise RuntimeError("KVMemoryManager already initialized")
            cls._instance = cls(model_name, kv_capacity, dtype)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("KVMemoryManager not initialized")
        return cls._instance