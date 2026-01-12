from pathlib import Path
import csv

class Query:
    def __init__(self, query: str, data_path: str):
        self.query = query
        self.data_path = data_path