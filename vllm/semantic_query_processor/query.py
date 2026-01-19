from pathlib import Path
import csv

class Query:
    def __init__(self, ops: str, data_path: str):
        self.ops = ops
        self.data_path = data_path