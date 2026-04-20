from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from vllm.semantic_query_processor.budget import KVMemoryManager
from vllm.semantic_query_processor.sem_ops import OpKind, ops


_TASK_IDS = count()


@dataclass
class Task:
    ctx: Any
    stage_index: int
    trackers: tuple = ()
    task_id: int = field(default_factory=lambda: next(_TASK_IDS))


class Stage:
    def __init__(
        self,
        stage_id: int,
        operators,
        kind: OpKind | None = None,
        fanout_op=None,
        priority_offset: int = 0,
    ):
        self.stage_id = stage_id
        self.operators = tuple(operators)
        self.kind = kind or self._infer_kind()
        self.fanout_op = fanout_op
        self.priority_offset = priority_offset
        self.waiting_tasks = deque()
        self.running_tasks = {}
        self.bytes_per_token = KVMemoryManager.get_instance().bytes_per_token
        self.low_watermark = 1
        self.high_watermark = 4

    def _infer_kind(self) -> OpKind:
        return OpKind.TUPLE_INDEPENDENT

    def enqueue(self, task: Task) -> None:
        self.waiting_tasks.append(task)

    def clear_tasks(self) -> None:
        self.waiting_tasks.clear()
        self.running_tasks.clear()

    def has_waiting_tasks(self) -> bool:
        return bool(self.waiting_tasks)

    def ready_count(self) -> int:
        return len(self.waiting_tasks)

    def running_count(self) -> int:
        return len(self.running_tasks)

    def is_starving(self) -> bool:
        return self.ready_count() < self.low_watermark

    def is_backlogged(self) -> bool:
        return self.ready_count() > self.high_watermark

    def tune_watermarks(self):
        if self.ready_count() == 0:
            self.low_watermark = max(1, self.low_watermark - 1)
            self.high_watermark = max(self.low_watermark + 1, self.high_watermark - 1)
        elif self.ready_count() > self.high_watermark:
            self.high_watermark += 1

    def peek_task(self) -> Task | None:
        if not self.waiting_tasks:
            return None
        return self.waiting_tasks[0]

    def pop_task(self) -> Task | None:
        if not self.waiting_tasks:
            return None
        return self.waiting_tasks.popleft()

    def estimate_budget(self, task: Task) -> int:
        max_boundary = -1
        for op in self.operators:
            if isinstance(op, ops.CartesianProduct):
                break

            if not hasattr(op, "estimate_tokens"):
                raise AttributeError(f"{op} must define `estimate_tokens`")

            estimated_tokens = op.estimate_tokens(task.ctx)
            if estimated_tokens > max_boundary:
                max_boundary = estimated_tokens

        return max_boundary * self.bytes_per_token

    def memory_limit(self, manager=None) -> int:
        manager = manager or KVMemoryManager.get_instance()
        _, limit = manager.stage_usage(self.stage_id)
        return limit

    async def can_accept(self, task: Task, manager=None) -> bool:
        manager = manager or KVMemoryManager.get_instance()
        return await manager.can_admit_stage(
            self.stage_id,
            self.estimate_budget(task),
        )

    async def accept(self, task: Task, manager=None) -> int:
        manager = manager or KVMemoryManager.get_instance()
        budget = self.estimate_budget(task)
        await manager.allocate_stage(self.stage_id, budget)
        self.running_tasks[task.task_id] = budget
        return budget

    async def force_accept(self, task: Task, manager=None) -> int:
        manager = manager or KVMemoryManager.get_instance()
        budget = self.estimate_budget(task)
        await manager.force_allocate_stage(self.stage_id, budget)
        self.running_tasks[task.task_id] = budget
        return budget

    async def release(self, task: Task, manager=None) -> None:
        manager = manager or KVMemoryManager.get_instance()
        budget = self.running_tasks.pop(task.task_id, 0)
        await manager.release_stage(self.stage_id, budget)

    async def run_task(self, task: Task):
        task.ctx.state.stage_id = self.stage_id

        keep_going = True
        for idx, op in enumerate(self.operators):
            if keep_going:
                keep_going = await op(
                    task.ctx,
                    priority=-(self.priority_offset + idx),
                )
                if keep_going is False:
                    return None

        return task.ctx


def stage_builder(operators, stage_id, priority_offset: int = 0):
    return Stage(
        stage_id=stage_id,
        operators=operators,
        priority_offset=priority_offset,
    )
