from .base import BaseOp, OpKind
from .endpoint import unpin_request, completion_call_internal


class SemFilter(BaseOp):
    def __init__(self, instruction, pin=False, is_last=False, max_len=10):   
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.instruction = instruction
        self.pin = pin
        self.is_last = is_last
        self.max_len = max_len

    async def __call__(self, ctx):
        prompt = (
            ctx.input.data
            + "\n\n"
            + self.instruction
            + "\n\n"
            + "Answer yes or no. Answer:"
        )

        res, req = await completion_call_internal(
            ctx.state.raw_request,
            prompt,
            self.max_len,
            pin=(self.pin and not self.is_last)
        )
        ctx.state.pin_req_id = res["id"]

        # if the answer is no, we can evict the pinned kv immediately
        if False:
            engine = ctx.state.raw_request.app.state.engine_client
            await unpin_request(engine, ctx.state.pin_req_id)

        return True
    

class SemMap(BaseOp):
    def __init__(self, instruction, pin=False, is_last=False, max_len=128):   
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.instruction = instruction
        self.pin = pin
        self.is_last = is_last
        self.max_len = max_len

    async def __call__(self, ctx):
        prompt = (
            ctx.input.data
            + "\n\n"
            + self.instruction
            + "\n\n"
        )

        res, req = await completion_call_internal(
            ctx.state.raw_request,
            prompt,
            self.max_len,
            pin=(self.pin and not self.is_last)
        )

        # sem_map always unpins the previous pin_req_id
        
        if ctx.state.pin_req_id is not None:
            
            engine = ctx.state.raw_request.app.state.engine_client
            await unpin_request(engine, ctx.state.pin_req_id)

        # TODO impl pin gen request if needed and update budget
        pass
        return res["choices"][0]["text"]


        
    

class SemGroupBy(BaseOp):
    def __init__(self, groups, pin=False, is_last=False, max_len=20):   
        self.kind = OpKind.TUPLE_INDEPENDENT
        self.groups = groups
        self.pin = pin
        self.is_last = is_last
        self.max_len = max_len

    async def __call__(self, ctx):
        prompt = (
            ctx.input.data
            + "\n\n"
            + self.groups
            + "\n\n"
            + "Choose the closest group. Answer:"
        )

        res, req = await completion_call_internal(
            ctx.state.raw_request,
            prompt,
            self.max_len,
            pin=(self.pin and not self.is_last)
        )
        ctx.state.pin_req_id = res["id"]

        # if last, it can evict the pinned kv immediately
        if self.is_last:
            engine = ctx.raw_request.app.state.engine_client
            await engine.engine_core.call_utility_async(
                "unpin_kv",
                ctx.state.pin_req_id,
            )

        return True


class SemAgg(BaseOp):
    pass


class SemTopK(BaseOp):
    def __init__(self, temp, pin=False, is_last=False, max_len=20):   
        self.kind = OpKind.TUPLE_DEPENDENT
        self.temp = temp
        self.pin = pin
        self.is_last = is_last
        self.max_len = max_len

    async def __call__(self, ctx):
        print('SemTopK called')
        return self.temp


class SemJoin(BaseOp):
    pass
