"""CPU test of WarmupFlatCooldown's min_lr (2026-11-07): two param groups at the d640 ladder's rates (Muon 3.4e-3, Adam 6.4e-4); warmup 1000,
flat to step 100200, linear cooldown to an absolute 3e-5 at step 133600; min_lr 0 keeps the old schedule (one lambda ending at 0)."""
import sys, torch
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "baselines/PixelGen")
from src.utils.lr_scheduler import WarmupFlatCooldown
def rates(min_lr, steps):
    a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([{"params": [a], "lr": 3.4e-3}, {"params": [b], "lr": 6.4e-4}], lr=1.0)
    sch = WarmupFlatCooldown(1000, 33400, 133600, min_lr)(opt)
    out, s = {}, 0
    for t in steps:
        while s < t:
            opt.step(); sch.step(); s += 1
        out[t] = [g["lr"] for g in opt.param_groups]
    return out
r = rates(3e-5, [500, 1000, 100200, 116900, 133600])
print(r)
assert abs(r[500][0] - 1.7e-3) < 1e-9 and abs(r[1000][0] - 3.4e-3) < 1e-9 and abs(r[100200][1] - 6.4e-4) < 1e-12
assert abs(r[133600][0] - 3e-5) < 1e-12 and abs(r[133600][1] - 3e-5) < 1e-12, r[133600]
assert abs(r[116900][0] - (3.4e-3 + 3e-5) / 2) < 1e-9 and abs(r[116900][1] - (6.4e-4 + 3e-5) / 2) < 1e-10
r0 = rates(0.0, [116900, 133600])
assert abs(r0[133600][0]) < 1e-15 and abs(r0[116900][0] - 1.7e-3) < 1e-9
print("ok")
