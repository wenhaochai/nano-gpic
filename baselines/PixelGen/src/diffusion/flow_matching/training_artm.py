"""The loss of ARTM on the final ladder (docs/artm_ladder.md; Transition Matching, Shaul et al. 2025, arXiv 2506.23589, section 2.2 and Algorithm 1).

Per image: a transition k uniform in {0, ..., T - 1}, two new noises e_k and e_{k+1}, X_k = (1 - k/T) e_k + (k/T) x and X_{k+1} = (1 - (k+1)/T) e_{k+1}
+ ((k+1)/T) x (the independent linear process; X_T = x). The model reads X_k's patches as the context and X_{k+1}'s patches shifted by one (teacher
forcing), and its head learns each patch of X_{k+1}: per patch a head time tau ~ logit-normal(head_P_mean, head_P_std) and noise eta,
q = tau patch + (1 - tau) eta, x-prediction, the official v-space L2 with t_eps, the mean over values (branch ar's and the DTM head's recipe).
"""
import torch

from src.diffusion.base.training import BaseTrainer


class ARTMTrainer(BaseTrainer):
    def __init__(self, head_P_mean=-0.8, head_P_std=1.6, t_eps=0.05, **kwargs):
        super().__init__(**kwargs)
        self.head_P_mean, self.head_P_std, self.t_eps = head_P_mean, head_P_std, t_eps

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        B = x.shape[0]
        T = net.art_T
        k = torch.randint(0, T, (B,), device=x.device)
        s_a = (k.float() / T).view(-1, 1, 1, 1)
        s_b = ((k + 1).float() / T).view(-1, 1, 1, 1)
        e_a, e_b = torch.randn_like(x), torch.randn_like(x)
        x_a = (1 - s_a) * e_a + s_a * x
        x_b = (1 - s_b) * e_b + s_b * x
        p = net.patch_size
        ctx = torch.nn.functional.unfold(x_a, kernel_size=p, stride=p).transpose(1, 2)
        tgt = torch.nn.functional.unfold(x_b, kernel_size=p, stride=p).transpose(1, 2)
        N = tgt.shape[1]
        tau = (torch.randn(B, N, device=x.device, dtype=torch.float32) * self.head_P_std + self.head_P_mean).sigmoid()
        eta = torch.randn_like(tgt)
        q = tau[..., None] * tgt + (1 - tau[..., None]) * eta
        if isinstance(metadata, dict) and metadata.get("y_pack") is not None:
            raise NotImplementedError("packed captions (lever E) run only the point model")
        y_mask = metadata.get("y_mask") if isinstance(metadata, dict) else None
        kw = {} if y_mask is None else {"y_mask": y_mask}
        hat = net(ctx, k.float() / T, y, art_tgt=tgt, art_q=q, art_tau=tau, **kw).float()
        per_value = ((hat - tgt.float()) / (1 - tau[..., None]).clamp_min(self.t_eps)) ** 2
        return dict(fm_loss=per_value.mean(), loss=per_value.mean(), loss_per_sample=per_value.mean(dim=(1, 2)))
