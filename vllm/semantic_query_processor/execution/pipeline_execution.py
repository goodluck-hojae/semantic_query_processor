import asyncio

from vllm.semantic_query_processor.budget import KVMemoryManager
from vllm.semantic_query_processor.controller.stage import Task
from vllm.semantic_query_processor.sem_ops import OpKind, ops


class PlanExecutor:

    def __init__(self):
        self.pipeline_executor = AsyncPipelineExecutor()
        self.blocking_executor = BlockingExecutor()

    async def execute(self, ctxs, plan):

        stage_stat_list = []
        for item in plan:

            # BLOCKING
            if isinstance(item, ops.BaseOp) and item.kind == OpKind.BLOCKING:
                ctxs = await item(ctxs)

            # STAGES
            else:
                ctxs, stage_stat = await self.pipeline_executor.execute_tasks(
                    ctxs,
                    item,
                )
                stage_stat_list.append(stage_stat)

        return ctxs, stage_stat_list


class JoinTracker:
    def __init__(
        self,
        parent_ctx,
        pending_children: int,
        manager: KVMemoryManager | None = None,
        stage_id: int | None = None,
        reserved_budget: int = 0,
    ):
        self.parent_ctx = parent_ctx
        self.pending_children = pending_children
        self.manager = manager
        self.stage_id = stage_id
        self.reserved_budget = reserved_budget
        self.lock = asyncio.Lock()

    async def child_finished(self):
        async with self.lock:
            self.pending_children -= 1
            should_finalize = self.pending_children == 0

        if not should_finalize:
            return

        if self.parent_ctx.state.pin_req_id is not None:
            await self.parent_ctx.state.executor.unpin(
                self.parent_ctx.state.raw_request,
                self.parent_ctx.state.pin_req_id,
            )
            self.parent_ctx.state.pin_req_id = None

        if (
            self.manager is not None
            and self.stage_id is not None
            and self.reserved_budget > 0
        ):
            await self.manager.release_stage(self.stage_id, self.reserved_budget)


class AsyncPipelineExecutor:
    def __init__(self):
        self.manager = KVMemoryManager.get_instance()

    async def execute_tasks(self, ctxs, stages):
        """
        ctxs: initial contexts
        stages: [stage1, stage2, ...]
        """
        stage_stat = {}
        out = []
        active = {}

        def log_running_tasks(event: str):
            stage_states = []
            for stage in stages:
                running_task_ids = sorted(stage.running_tasks.keys())
                waiting_task_ids = [task.task_id for task in stage.waiting_tasks]
                used, cap = self.manager.stage_usage(stage.stage_id)
                stage_states.append(
                    "stage="
                    f"{stage.stage_id} "
                    f"used={used:,} cap={cap:,} "
                    f"waiting={len(waiting_task_ids)} "
                    f"running={len(running_task_ids)}"
                )
            print(f"[scheduler] {event} | " + " | ".join(stage_states))

        async def finalize_task(task: Task):
            for tracker in task.trackers:
                await tracker.child_finished()

        def record_input(stage_id: int):
            if stage_id in stage_stat:
                stage_stat[stage_id]["input"] += 1
            else:
                stage_stat[stage_id] = {"input": 1, "output": 0}

        def record_output(stage_id: int, count: int):
            if stage_id not in stage_stat:
                stage_stat[stage_id] = {"input": 0, "output": 0}
            stage_stat[stage_id]["output"] += count

        def has_pending_work():
            return any(stage.has_waiting_tasks() for stage in stages)

        async def rebalance_stage_capacity():
            changed = False

            for stage in stages:
                stage.tune_watermarks()

            # Keep draining idle stages toward their floor so later stages
            # can absorb capacity when they are the bottleneck.
            for stage in stages:
                if stage.ready_count() == 0 and stage.running_count() == 0:
                    while await self.manager.return_stage_capacity(stage.stage_id):
                        changed = True

            # Grow any stage with a real backlog. Prefer borrowing from the
            # immediate upstream stage first because that matches pipeline flow.
            for idx, stage in enumerate(stages):
                if not stage.is_backlogged():
                    continue

                donor_hint = stages[idx - 1].stage_id if idx > 0 else None
                while await self.manager.rebalance_stage_capacity(
                    receiver_id=stage.stage_id,
                    donor_hint=donor_hint,
                ):
                    changed = True

            # If a later stage is empty while the previous stage still has work,
            # bias capacity upstream so the pipeline can produce tuples for it.
            for idx in range(1, len(stages)):
                stage = stages[idx]
                prev_stage = stages[idx - 1]
                if stage.is_starving() and prev_stage.has_waiting_tasks():
                    while await self.manager.rebalance_stage_capacity(
                        receiver_id=prev_stage.stage_id,
                        donor_hint=stage.stage_id,
                    ):
                        changed = True

            if changed:
                log_running_tasks("rebalance")

        def blocked_stage_details():
            details = {}
            for stage in stages:
                used, cap = self.manager.stage_usage(stage.stage_id)
                head_task = stage.peek_task()
                head_budget = None
                head_task_id = None
                admissible = None
                if head_task is not None:
                    head_task_id = head_task.task_id
                    head_budget = stage.estimate_budget(head_task)
                    admissible = used + head_budget <= cap
                details[stage.stage_id] = {
                    "used": used,
                    "cap": cap,
                    "waiting": len(stage.waiting_tasks),
                    "running": len(stage.running_tasks),
                    "head_task_id": head_task_id,
                    "head_budget": head_budget,
                    "head_admissible": admissible,
                }
            return details

        async def launch_ready_tasks():
            launched = False

            for stage in stages:
                while stage.has_waiting_tasks():
                    task = stage.peek_task()
                    if task is None:
                        break

                    if not await stage.can_accept(task, self.manager):
                        break

                    record_input(stage.stage_id)
                    stage.pop_task()
                    await stage.accept(task, self.manager)
                    log_running_tasks(
                        f"start task={task.task_id} stage={stage.stage_id}"
                    )
                    future = asyncio.create_task(stage.run_task(task))
                    active[future] = (stage, task)
                    launched = True

            return launched

        async def force_launch_blocked_tasks():
            launched = False

            for stage in stages:
                task = stage.peek_task()
                if task is None:
                    continue

                record_input(stage.stage_id)
                stage.pop_task()
                await stage.force_accept(task, self.manager)
                log_running_tasks(
                    f"force-start task={task.task_id} stage={stage.stage_id}"
                )
                future = asyncio.create_task(stage.run_task(task))
                active[future] = (stage, task)
                launched = True

            return launched

        for stage in stages:
            stage.clear_tasks()

        if stages:
            for ctx in ctxs:
                stages[0].enqueue(Task(ctx=ctx, stage_index=0))

        while has_pending_work() or active:
            await rebalance_stage_capacity()
            await launch_ready_tasks()

            if not active:
                if has_pending_work():
                    forced = await force_launch_blocked_tasks()
                    if not forced:
                        raise RuntimeError(
                            "No runnable tasks remain, but stage queues are not empty. "
                            f"Blocked stage details: {blocked_stage_details()}"
                        )
                    continue
                break

            done, _ = await asyncio.wait(
                set(active.keys()),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for future in done:
                stage, task = active.pop(future)
                budget = stage.detach_budget(task)
                try:
                    result = future.result()
                except Exception:
                    await stage.release_budget(budget, self.manager)
                    log_running_tasks(
                        f"finish task={task.task_id} stage={stage.stage_id}"
                    )
                    raise

                if result is None:
                    await stage.release_budget(budget, self.manager)
                    log_running_tasks(
                        f"finish task={task.task_id} stage={stage.stage_id}"
                    )
                    await rebalance_stage_capacity()
                    await finalize_task(task)
                    continue

                next_stage_index = task.stage_index + 1
                if stage.fanout_op is not None:
                    child_ctxs = stage.fanout_op(result) or []
                    record_output(stage.stage_id, len(child_ctxs))

                    if not child_ctxs:
                        if task.ctx.state.pin_req_id is not None:
                            await task.ctx.state.executor.unpin(
                                task.ctx.state.raw_request,
                                task.ctx.state.pin_req_id,
                            )
                            task.ctx.state.pin_req_id = None
                        await stage.release_budget(budget, self.manager)
                        log_running_tasks(
                            f"finish task={task.task_id} stage={stage.stage_id}"
                        )
                        await rebalance_stage_capacity()
                        await finalize_task(task)
                        continue

                    deferred_release = False
                    trackers = task.trackers
                    reserved_budget = 0
                    if task.ctx.state.pin_req_id is not None:
                        reserved_budget = min(
                            budget,
                            stage.estimate_pinned_budget(task),
                        )
                        immediate_release = budget - reserved_budget
                        if immediate_release > 0:
                            await self.manager.release_stage(
                                stage.stage_id,
                                immediate_release,
                                decrement_inflight=False,
                            )
                        trackers = trackers + (
                            JoinTracker(
                                task.ctx,
                                len(child_ctxs),
                                manager=self.manager,
                                stage_id=stage.stage_id,
                                reserved_budget=reserved_budget,
                            ),
                        )
                        deferred_release = reserved_budget > 0

                    for child_ctx in child_ctxs:
                        child_task = Task(
                            ctx=child_ctx,
                            stage_index=next_stage_index,
                            trackers=trackers,
                        )
                        if next_stage_index >= len(stages):
                            out.append(child_ctx)
                            await finalize_task(child_task)
                        else:
                            stages[next_stage_index].enqueue(child_task)

                    if not deferred_release:
                        await stage.release_budget(budget, self.manager)

                    log_running_tasks(
                        f"finish task={task.task_id} stage={stage.stage_id} "
                        f"deferred_release={deferred_release}"
                    )
                    await rebalance_stage_capacity()
                    continue

                record_output(stage.stage_id, 1)
                next_task = Task(
                    ctx=result,
                    stage_index=next_stage_index,
                    trackers=task.trackers,
                )
                if next_stage_index >= len(stages):
                    out.append(result)
                    await finalize_task(next_task)
                else:
                    stages[next_stage_index].enqueue(next_task)

                await stage.release_budget(budget, self.manager)
                log_running_tasks(
                    f"finish task={task.task_id} stage={stage.stage_id}"
                )
                await rebalance_stage_capacity()

        return out, stage_stat


class BlockingExecutor:

    @staticmethod
    async def execute_tasks(
        seeds,
        task_builder,
        concurrency: int = 100,
    ):
        manager = KVMemoryManager.get_instance()

        queue = asyncio.Queue(maxsize=concurrency)
        capacity_cond = asyncio.Condition()
        results = []

        async def worker():
            while True:
                task = await queue.get()
                try:
                    out = await task()
                    results.append(out)
                finally:
                    async with capacity_cond:
                        await manager.release(task.budget)
                        capacity_cond.notify_all()
                    queue.task_done()

        workers = [
            asyncio.create_task(worker())
            for _ in range(concurrency)
        ]

        for seed in seeds:
            task = task_builder(seed)

            async with capacity_cond:
                while not await manager.can_admit(task.budget):
                    await capacity_cond.wait()
                await manager.allocate(task.budget)

            await queue.put(task)

        await queue.join()

        for w in workers:
            w.cancel()

        return results
