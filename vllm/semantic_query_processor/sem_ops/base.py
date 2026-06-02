from enum import Enum, auto
from dataclasses import dataclass, field


class OpName:
    SEM_FILTER = "sem_filter"
    SEM_JOIN = "sem_join"
    SEM_CLASSIFY = "sem_classify"
    SEM_TOPK = "sem_topk"
    SEM_MAP = "sem_map"
    SEM_AGG = "sem_agg"
    JOIN = "join"
    CARTESIAN_PRODUCT = "cp"
    INDEXED_SEARCH = "indexed_search"


OPERATOR_LIST = [
    OpName.SEM_FILTER,
    OpName.SEM_JOIN,
    OpName.SEM_CLASSIFY,
    OpName.SEM_TOPK,
    OpName.SEM_MAP,
    OpName.SEM_AGG,
    OpName.JOIN,
    OpName.CARTESIAN_PRODUCT,
    OpName.INDEXED_SEARCH,
]


class OpKind(Enum):
    TUPLE_INDEPENDENT = auto()   # sem_filter, sem_map, sem_classify, sem_join
    BLOCKING = auto()     # sem_topk, sem_agg, 
    JOIN = auto()     # join, cartesian product


class BaseOp:
    def __init__(self, kind, position):
        self.kind = kind
        self.position = position

    async def __call__(self, ctx):
        raise NotImplementedError
     
