"""The loss of the DTM flow head on the final ladder (docs/dtm_ladder.md; Transition Matching, Shaul et al. 2025, arXiv 2506.23589, section 2.2).

The backbone's process is the point model's (JiTTrainer): t from the same distribution (logit-normal, P_mean -0.8, P_std 0.8, the same shift),
x_t = t x + (1 - t) e, one t and one e per image, drawn in the same order. The point model regresses the mean of p(x | x_t, y); here a small flow head
learns the whole distribution of each patch given the backbone's token at that patch (DTM's p(Y | X_t) with Y = x - e, a bijection of x given x_t).
The head's recipe is branch ar's (training_ar.py): per patch its own head time tau ~ logit-normal(head_P_mean, head_P_std) and noise eta,
q = tau patch + (1 - tau) eta, x-prediction, and the official v-space L2 with t_eps: ((x_hat - patch) / max(1 - tau, t_eps))^2, the mean over
values. The paper samples t uniformly over 32 discrete steps; the ladder's t distribution is kept, so that the head is the only change.
"""
import torch

from src.diffusion.flow_matching.training_JiT import JiTTrainer, time_shift_fn


class DTMTrainer(JiTTrainer):
    def __init__(self, *args, head_P_mean=-0.8, head_P_std=1.6, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.lognorm_t, "the ladder's t distribution (lognorm_t) is the only one tested"
        self.head_P_mean, self.head_P_std = head_P_mean, head_P_std

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        B = x.shape[0]
        base_t = (torch.randn(B, device=x.device, dtype=torch.float32) * self.P_std + self.P_mean).sigmoid()   # JiTTrainer's draws, in its order
        t = time_shift_fn(base_t, self.timeshift)
        noise = torch.randn_like(x)
        x_t = self.scheduler.alpha(t) * x + noise * self.scheduler.sigma(t)
        p = net.patch_size
        patches = torch.nn.functional.unfold(x, kernel_size=p, stride=p).transpose(1, 2)   # [B, N, C p p], the model's own token layout
        N = patches.shape[1]
        tau = (torch.randn(B, N, device=x.device, dtype=torch.float32) * self.head_P_std + self.head_P_mean).sigmoid()
        eta = torch.randn_like(patches)
        q = tau[..., None] * patches + (1 - tau[..., None]) * eta
        y_mask = metadata.get("y_mask") if isinstance(metadata, dict) else None
        if isinstance(metadata, dict) and metadata.get("y_pack") is not None:
            raise NotImplementedError("packed captions (lever E) run only the point model")
        kw = {} if y_mask is None else {"y_mask": y_mask}
        hat = net(x_t, t, y, dtm_q=q, dtm_tau=tau, **kw).float()
        per_value = ((hat - patches.float()) / (1 - tau[..., None]).clamp_min(self.t_eps)) ** 2
        return dict(fm_loss=per_value.mean(), loss=per_value.mean(), loss_per_sample=per_value.mean(dim=(1, 2)))
