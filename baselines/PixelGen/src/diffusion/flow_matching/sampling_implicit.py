"""Sampler of the implicit model. If x_hat ~ p(x | x_t), then x_t + dt / (1 - t) (x_hat - x_t) = (t + dt) x_hat + (1 - t - dt) eps_hat, with
eps_hat = (x_t - t x_hat) / (1 - t), and (x_hat, eps_hat) has the law of (x, eps) given x_t: the next state has the right marginal at any step
size, and the last step returns x_hat. So there is no t_eps here. Errors come only from the model. num_steps is the number of network calls.
"""
import torch

from src.diffusion.flow_matching.sampling import EulerSamplerJiT


class ImplicitSampler(EulerSamplerJiT):
    def _impl_sampling(self, net, noise, condition, uncondition):
        assert float(self.guidance) == 1.0, "CFG is fixed at 1.0 (owner, 2026-11-01), and it has no meaning here yet"
        steps = self.timesteps.to(noise.device, noise.dtype)
        x = noise
        x_trajs, v_trajs = [noise], []
        for t_cur, t_next in zip(steps[:-1], steps[1:]):
            t = t_cur.repeat(noise.shape[0])
            v = (net(x, t, condition) - x) / (1 - t_cur)
            x = x + (t_next - t_cur) * v
            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs
