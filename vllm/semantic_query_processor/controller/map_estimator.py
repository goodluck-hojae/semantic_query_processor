import numpy as np
from collections import defaultdict


class MapRatioEstimator:
    _instance = None

    def __init__(self, sample_size: int):
        self.data = defaultdict(list)
        self.minimum_sample_size = sample_size

    @classmethod
    def instance(cls, sample_size: int = 10):
        if cls._instance is None:
            cls._instance = cls(sample_size)
        return cls._instance

    def update(self, pos, input_len, output_len):
        if input_len <= 0:
            return

        ratio = output_len / input_len
        self.data[pos].append(ratio)

    def get_ratio(self, pos, percentile=95):
        if pos not in self.data:
            return None

        if len(self.data[pos]) < self.minimum_sample_size:
            return None

        return np.percentile(self.data[pos], percentile)
    
    def reset(self):
        self.data.clear()