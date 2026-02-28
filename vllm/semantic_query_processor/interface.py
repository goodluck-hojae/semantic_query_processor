from pydantic import BaseModel
from pathlib import Path
import csv


class SemanticQueryRequest(BaseModel):
    ops: list
    data_path: str