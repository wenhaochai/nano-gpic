"""Does a graph compiled OUTSIDE a DDP forward (as CompileWarmup does: pl_module.denoiser called directly) get reused
inside the DDP forward, skipping dynamo's DDPOptimizer (graph split at bucket boundaries, so the gradient all-reduce
overlaps the compiled backward)? CPU, gloo, world 1, tiny bucket cap to force splits. Counts DDPOptimizer invocations."""
import os, torch, torch.nn as nn, torch.distributed as dist
from torch._dynamo.backends import distributed as dd
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533")
dist.init_process_group("gloo", rank=0, world_size=1)
calls = []
orig = dd.DDPOptimizer.compile_fn
def spy(self, gm, inputs):
    calls.append(len(list(gm.graph.nodes))); return orig(self, gm, inputs)
dd.DDPOptimizer.compile_fn = spy

def make():
    torch.manual_seed(0)
    inner = nn.Sequential(*[nn.Linear(256, 256) for _ in range(6)])
    inner.compile(backend="aot_eager")
    class Wrap(nn.Module):           # stands in for the LightningModule that DDP wraps
        def __init__(s): super().__init__(); s.inner = inner
        def forward(s, x): return s.inner(x)
    w = Wrap()
    return w, nn.parallel.DistributedDataParallel(w, bucket_cap_mb=0.1)

for warm in (False, True):
    torch._dynamo.reset(); calls.clear()
    w, ddp = make()
    if warm:   # CompileWarmup: the inner module called directly, before any DDP forward
        w.inner(torch.randn(32, 256)).sum().backward(); w.zero_grad(set_to_none=True)
    n0 = len(calls)
    for _ in range(3):
        ddp(torch.randn(32, 256)).sum().backward()
    print(f"warmup={warm}: DDPOptimizer.compile_fn calls during warm-up {n0}, during 3 DDP steps {len(calls) - n0}")
dist.destroy_process_group()
