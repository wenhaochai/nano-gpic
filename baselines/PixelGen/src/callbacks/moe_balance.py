"""MoE load balance and monitoring (branch moe; src/models/layers/moe.py). Once per optimizer step, before the step:
  1. every MoE layer's statistics of the step (moe_stat: load per expert, boundary score per expert, dropped assignments,
     tokens, calls; summed over the accumulated micro-batches) are summed over the ranks with one all_reduce;
  2. the routing state is updated, the same on every rank: route tc: moe_bias += rate * sign(mean load - load) (DeepSeek's
     loss-free balancing, 2408.15664); route ecb: moe_thr, the evaluation threshold, follows the mean boundary score (EMA);
  3. the statistics are zeroed.
Every log_every steps rank 0 logs, per layer and over the window: MaxVio = max load / mean load - 1 (0 = balanced), the
normalized entropy of the load, the share of experts with no token, and the share of dropped assignments (route tc).
"""
import math

import torch
import torch.distributed as dist
from lightning.pytorch import Callback


class MoEBalance(Callback):
    def __init__(self, log_every: int = 100, thr_decay: float = 0.99):
        super().__init__()
        self.log_every, self.thr_decay = log_every, thr_decay
        self.layers, self.names, self._win, self._steps = [], [], None, 0

    def setup_layers(self, net):
        from src.models.layers.moe import MoEFeedForward
        self.layers, self.names = [], []
        for name, m in net.named_modules():
            if isinstance(m, MoEFeedForward):
                self.layers.append(m)
                self.names.append(name.replace("blocks.", "b").replace(".mlp", ""))
        assert len({m.n for m in self.layers}) <= 1, "MoE layers of one model have one expert count"

    @torch.no_grad()
    def update(self):
        """Steps 1 to 3. Returns the summed statistics [layers, 2n + 3] (None without MoE layers)."""
        if not self.layers:
            return None
        stat = torch.stack([m.moe_stat for m in self.layers])
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(stat)
        # No host sync: the branches on GPU tensors are torch.where (2026-11-14, step profile w2686: `if calls > 0` and
        # `bool(moe_thr_set)` synced 24 times a step at d12, the first one waiting for the whole backward, so the optimizer
        # and the next forward ran host-bound). The same values bit for bit (tools/moe/tests/test_balance_nosync.py).
        for m, s in zip(self.layers, stat):
            n = m.n
            load, thr, calls = s[:n], s[n:2 * n], s[2 * n + 2]
            on = calls > 0
            if m.route == "tc":
                m.moe_bias.copy_(torch.where(on, m.moe_bias + m.bias_rate * torch.sign(load.mean() - load), m.moe_bias))
            elif m.route == "ecb":
                mean = thr / calls.clamp_min(1)
                m.moe_thr.copy_(torch.where(on, torch.where(m.moe_thr_set, torch.lerp(m.moe_thr, mean, 1 - self.thr_decay), mean), m.moe_thr))
                m.moe_thr_set.logical_or_(on)
            m.moe_stat.zero_()
        return stat

    @staticmethod
    def summary(s, n, G):
        """One layer's window statistics -> {maxvio, entropy, dead, drop}."""
        load, drops, tokens = s[:n], s[2 * n], s[2 * n + 1]
        tot = load.sum().clamp_min(1)
        p = load / tot
        ent = -(p * torch.log(p.clamp_min(1e-30))).sum() / math.log(n) if n > 1 else torch.ones(())
        return {"maxvio": (load.max() / load.mean().clamp_min(1e-30) - 1).item(), "entropy": float(ent),
                "dead": (load == 0).float().mean().item(), "drop": (drops / tot).item()}

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        stat = self.update()
        if stat is None:
            return
        self._win = stat.clone() if self._win is None else self._win + stat
        self._steps += 1
        if self._steps % self.log_every == 0:
            if trainer.is_global_zero and trainer.logger is not None:
                out, worst = {}, {"maxvio": 0.0, "dead": 0.0, "drop": 0.0, "entropy": 1.0}
                for name, m, s in zip(self.names, self.layers, self._win):
                    r = self.summary(s, m.n, m.G)
                    for k, v in r.items():
                        out[f"moe/{k}_{name}"] = v
                    for k in ("maxvio", "dead", "drop"):
                        worst[k] = max(worst[k], r[k])
                    worst["entropy"] = min(worst["entropy"], r["entropy"])
                out.update({f"moe/{k}_worst": v for k, v in worst.items()})
                trainer.logger.log_metrics(out, step=trainer.global_step)
                print(f"[moe] step {trainer.global_step}: worst layer over {self.log_every} steps: MaxVio {worst['maxvio']:.3f}, "
                      f"entropy {worst['entropy']:.4f}, dead {worst['dead']:.3f}, dropped {worst['drop']:.4f}", flush=True)
            self._win = None
