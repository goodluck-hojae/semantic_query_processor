from enum import Enum, auto
from dataclasses import dataclass, field


class OpName:
    SEM_FILTER = "sem_filter"
    SEM_JOIN = "sem_join"
    SEM_CLASSIFY = "sem_classify"
    LEGACY_SEM_GROUPBY = "sem_groupby"
    SEM_TOPK = "sem_topk"
    SEM_MAP = "sem_map"
    SEM_AGG = "sem_agg"
    JOIN = "join"


OPERATOR_LIST = [
    OpName.SEM_FILTER,
    OpName.SEM_JOIN,
    OpName.SEM_CLASSIFY,
    OpName.LEGACY_SEM_GROUPBY,
    OpName.SEM_TOPK,
    OpName.SEM_MAP,
    OpName.SEM_AGG,
    OpName.JOIN,
]


class BaseOp:
    max_len: int

    async def __call__(self, ctx):
        raise NotImplementedError
     
class OpKind(Enum):
    TUPLE_INDEPENDENT = auto()   # sem_filter, sem_map, sem_classify, sem_join
    BLOCKING = auto()     # sem_topk, sem_agg, 
    JOIN = auto()     # join, cartesian product
