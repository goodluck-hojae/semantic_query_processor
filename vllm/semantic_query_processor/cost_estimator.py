import csv
from pathlib import Path

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


class KVEstimator:
    def __init__(self, model_name: str, kv_capacity: int, dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bytes_per_token = compute_bytes_per_token(model_name, dtype)
        self.kv_capacity = kv_capacity * 0.95
    
    def token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def can_admit(self, budget: int) -> bool:
        return budget < self.kv_capacity 

    def allocate(self, budget: int):
        # print(f"allocate remaining={self.kv_capacity/1024**3:6.2f}GB")
        if not self.can_admit(budget):
            return False
        self.kv_capacity -= budget
        return True
        
    # When completed
    def release(self, budget: int):
        # print(f"release remaining={self.kv_capacity/1024**3:6.2f}GB")
        self.kv_capacity += budget
        return True


if __name__ == "__main__":
    MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    KV_CAPACITY_BYTES = 7117927424

    kv = KVEstimator(MODEL, KV_CAPACITY_BYTES)

    print("Bytes per token:", kv.bytes_per_token)
    print("Initial KV capacity (GB):", kv.kv_capacity / 1024**3)

    path = Path(
        "/home/hojaeson_umass_edu/.cache/kagglehub/datasets/"
        "snehaanbhawal/resume-dataset/versions/1/Resume/Resume.csv"
    )

    admitted = 0
    rejected = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            resume = row["Resume_str"].strip()

            tokens = kv.count_tokens(resume)
            kv_bytes = tokens * kv.bytes_per_token

            ok = kv.add_request(resume)

            if ok:
                admitted += 1
            else:
                rejected += 1

            print(
                f"[{i:03d}] tokens={tokens:<5} "
                f"req_kv={kv_bytes/1024**2:6.1f}MB "
                f"remaining={kv.kv_capacity/1024**3:6.2f}GB "
                f"status={'ADMIT' if ok else 'REJECT'}"
            )

            if i == 200:
                break

    print("\n===== SUMMARY =====")
    print("Admitted:", admitted)
    print("Rejected:", rejected)
    print("Remaining KV capacity (GB):", kv.kv_capacity / 1024**3)