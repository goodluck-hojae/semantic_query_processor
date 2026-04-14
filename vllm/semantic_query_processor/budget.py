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
    LOG = False
    def __init__(self, model_name, kv_capacity, dtype):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bytes_per_token = compute_bytes_per_token(model_name, dtype)

        self._capacity = kv_capacity 
        self._global_used = 0

        self._stage_capacity = {}
        self._stage_used = {}
        self._stage_inflight = {}

        self._cond = asyncio.Condition()

    # Stage API (Pipeline)
    def register_stage(self, stage_id: int, fraction: float):
        self._stage_capacity[stage_id] = self._capacity * fraction
        self._stage_used[stage_id] = 0
        self._stage_inflight[stage_id] = 0

    async def can_admit_stage(self, stage_id: int, budget: int):
        async with self._cond:
            return (
                self._stage_used[stage_id] + budget
                <= self._stage_capacity[stage_id]
            )

    async def allocate_stage(self, stage_id: int, budget: int):
        if KVMemoryManager.LOG:
            used, cap = self.stage_usage(stage_id)
            print("stage", stage_id, "inflight", self.stage_inflight(stage_id), "used", used, "cap", cap, "budget", budget)

        async with self._cond:
            while (
                self._stage_used[stage_id] + budget
                > self._stage_capacity[stage_id]
            ):
                await self._cond.wait()

            self._stage_used[stage_id] += budget
            self._stage_inflight[stage_id] += 1

    async def release_stage(self, stage_id: int, budget: int):
        if KVMemoryManager.LOG:
            used, cap = self.stage_usage(stage_id)
            print("stage", stage_id, "inflight", self.stage_inflight(stage_id), "used", used, "cap", cap, "budget", budget)

        async with self._cond: 
            self._stage_used[stage_id] -= budget
            if self._stage_used[stage_id] < 0:
                self._stage_used[stage_id] = 0
            self._stage_inflight[stage_id] -= 1
            if self._stage_inflight[stage_id] < 0:
                self._stage_inflight[stage_id] = 0

            self._cond.notify_all()

    async def wait_for_stage_capacity(self):
        async with self._cond:
            await self._cond.wait()
            
    def stage_inflight(self, stage_id: int) -> int:
        return self._stage_inflight.get(stage_id, 0)

    def stage_usage(self, stage_id: int) -> tuple[int, int]:
        return (
            int(self._stage_used.get(stage_id, 0)),
            int(self._stage_capacity.get(stage_id, 0)),
        )

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


    def token_length(self, text) -> int:
        if type(text) is str:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            prompt = self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=False)
            return len(self.tokenizer.encode(prompt, add_special_tokens=False))

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
    
    def apply_chat_template(self, prompt):
        return self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
    
    def capacity(self):
        return self._capacity
