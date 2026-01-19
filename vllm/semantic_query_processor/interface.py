from pydantic import BaseModel
from pathlib import Path
import csv


class SemanticQueryRequest(BaseModel):
    ops: str
    data_path: str