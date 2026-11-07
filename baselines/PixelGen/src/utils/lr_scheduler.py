# from transformers import get_constant_schedule_with_warmup

# class ConstantWithWarmup:
#     def __init__(self, num_warmup_steps: int):
#         self.num_warmup_steps = num_warmup_steps

#     def __call__(self, optimizer):
#         return get_constant_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=self.num_warmup_steps
#         )


from transformers import get_constant_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

class ConstantWithWarmup:
    def __init__(self, num_warmup_steps: int):
        self.num_warmup_steps = num_warmup_steps

    def __call__(self, optimizer):
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps
        )

class WarmupFlatCooldown:
    """Linear warmup, a flat rate, then a linear cooldown over the last num_cooldown_steps of max_steps.
    min_lr (owner 2026-11-07: 「学习率线性不降到0，而是降低到一个固定值3e-5」): the absolute rate every param group reaches at
    max_steps; each group gets its own lambda, ending at min_lr / its initial rate (the Muon and the Adam groups start at
    different rates). min_lr = 0 keeps the old schedule exactly (one lambda, down to 0)."""
    def __init__(self, num_warmup_steps: int, num_cooldown_steps: int, max_steps: int, min_lr: float = 0.0):
        self.num_warmup_steps = num_warmup_steps
        self.num_cooldown_steps = num_cooldown_steps
        self.max_steps = max_steps
        self.min_lr = min_lr

    def __call__(self, optimizer):
        warmup = self.num_warmup_steps
        cooldown = self.num_cooldown_steps
        total = self.max_steps

        def make(floor):
            def lr_lambda(step):
                if step < warmup:
                    return step / max(1, warmup)
                elif step < total - cooldown:
                    return 1.0
                else:
                    frac = min(1.0, (step - (total - cooldown)) / max(1, cooldown))
                    return 1.0 - (1.0 - floor) * frac
            return lr_lambda

        if not self.min_lr:
            return LambdaLR(optimizer, lr_lambda=make(0.0))
        floors = [min(1.0, self.min_lr / g.get("initial_lr", g["lr"])) for g in optimizer.param_groups]
        return LambdaLR(optimizer, lr_lambda=[make(f) for f in floors])
        # return get_constant_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.num_warmup_steps
        # )