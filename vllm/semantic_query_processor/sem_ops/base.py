from enum import Enum, auto
from dataclasses import dataclass, field


OPERATOR_LIST = ['sem_filter', 'sem_join', 'sem_groupby', 'sem_topk', 'sem_map', 'sem_agg']


class BaseOp:
    max_len: int
    is_last: bool = False

    async def __call__(self, ctx):
        raise NotImplementedError


class OpKind(Enum):
    TUPLE_INDEPENDENT = auto()   # sem_filter, sem_map, sem_groupby, sem_join
    TUPLE_DEPENDENT = auto()     # sem_topk, sem_agg, 

