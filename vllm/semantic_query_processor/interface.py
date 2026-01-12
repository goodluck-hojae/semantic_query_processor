from pydantic import BaseModel
from pathlib import Path
import csv


class SemanticQueryRequest(BaseModel):
    query: str
    data_path: str