from dataclasses import dataclass


@dataclass
class KVConfig:
    bytes_per_token: int
    max_classifiy_tokens: int = 10    # boolean answer
    max_gen_tokens: int = 2048


class KVEstimator:
    """
    Runtime KV usage estimator.
    Tracks cumulative KV usage for active tuples.
    """

    def __init__(self, kv_config: KVConfig, max_kv_bytes: int):
        self.cfg = kv_config
        self.max_kv_bytes = max_kv_bytes
        self.current_kv_bytes = 0

    # ----------------------------
    # Per-tuple estimation
    # ----------------------------

    def estimate_filter_cost(self, input_tokens: int) -> int:
        """
        KV cost after sem_filter.
        """
        return (
            (input_tokens + self.cfg.max_classifiy_tokens) * self.cfg.bytes_per_token
        )

    def estimate_map_cost(self) -> int:
        """
        Additional KV cost introduced by sem_map.
        """
        return self.cfg.max_gen_tokens * self.cfg.bytes_per_token

    def estimate_total_tuple_cost(self, input_tokens: int) -> int:
        """
        Total KV footprint if tuple reaches sem_map.
        """
        return (
            self.estimate_filter_cost(input_tokens)
            + self.estimate_map_cost()
        )

    # ----------------------------
    # Admission control
    # ----------------------------
    def can_admit(self, input_tokens: int) -> bool:
        """
        Check whether a new tuple can fit in KV cache.
        """
        print('can_admit called')
        est = self.estimate_total_tuple_cost(input_tokens)
        return self.current_kv_bytes + est <= self.max_kv_bytes


    def admit(self, input_tokens: int):
        """
        Admit tuple and charge its KV usage.
        """
        print('admit called')
        est = self.estimate_total_tuple_cost(input_tokens)
        self.current_kv_bytes += est

    # ----------------------------
    # Eviction / boundary

    # ----------------------------
    def reset(self):
        """
        Called at semantic boundary.
        """
        self.current_kv_bytes = 0


if __name__ == "__main__":
    cfg = KVConfig(
        bytes_per_token=16,
        filter_gen_tokens=2,
        map_max_gen_tokens=2048,
    )

    kv_estimator = KVEstimator(
        kv_config=cfg,
        max_kv_bytes=8 * 1024 * 1024 * 1024,  # 8 GB
    )

    n_tokens_list = [1000, 5000, 20000, 100000, 500000, 1000000, 1000, 5000, 20000, 100000, 500000, 1000000, 1000, 5000, 20000, 100000, 500000, 1000000, 1000, 5000, 20000, 100000, 500000, 1000000]  * 300

    for idx, n_tokens in enumerate(n_tokens_list):
        if kv_estimator.can_admit(n_tokens):
            kv_estimator.admit(n_tokens)
        else:
            break
        print(idx, kv_estimator.current_kv_bytes, kv_estimator.max_kv_bytes)
        
    op1 = sem_filter
    op2 = sem_map
    for idx, input in enumerate(n_tokens_list):
        is_true = op1(input)
        if is_true:
            if kv_estimator.can_admit(input):
                kv_estimator.admit(input)
                current_batch.append(input)
            else:
                op2(current_batch)
            


