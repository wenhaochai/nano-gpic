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
    def __init__(self, num_warmup_steps: int, num_cooldown_steps: int, max_steps: int):
        self.num_warmup_steps = num_warmup_steps
        self.num_cooldown_steps = num_cooldown_steps
        self.max_steps = max_steps

    def __call__(self, optimizer):
        warmup = self.num_warmup_steps
        cooldown = self.num_cooldown_steps
        total = self.max_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            elif step < total - cooldown:
                return 1.0
            else:
                steps_into_decay = step - (total - cooldown)
                return 1.0 - steps_into_decay / max(1, cooldown)

        return LambdaLR(optimizer, lr_lambda=lr_lambda)
        # return get_constant_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.num_warmup_steps
        # )