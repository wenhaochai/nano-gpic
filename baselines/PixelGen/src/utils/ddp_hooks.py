"""DDP communication hooks for the speed campaign."""
import torch.distributed as dist


def allreduce_avg_hook(state, bucket):
    """Average in the collective (NCCL ReduceOp.AVG, a pre-multiplied sum by 1/world) instead of the reducer's
    per-parameter multiply by 1/world before a SUM all-reduce (perf C8: one elementwise pass over all gradients,
    3.9 ms per step on 8 H100s). The world size is a power of two here, so the scaling is exact either way."""
    fut = dist.all_reduce(bucket.buffer(), op=dist.ReduceOp.AVG, group=state, async_op=True).get_future()
    return fut.then(lambda f: f.value()[0])
