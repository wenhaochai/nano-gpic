"""The DTM sampler on the final ladder (docs/dtm_ladder.md; Transition Matching, Shaul et al. 2025, arXiv 2506.23589, Algorithm 2): at every call
the backbone runs once, the flow head draws each patch of x_hat from p(patch | x_t, y) by its probabilities (net(..., dtm_mode="draw")), and the
state takes the straight step along the draw's velocity, x' = x + (sigma - sigma') (x_hat - x) / sigma, with no fresh noise; the last step returns
the draw. This is gm_draw (sampling_gmflow.GMDrawSampler) with the flow head in place of the per-pixel mixture, and DTM's X_{t+1} = X_t + Y / T
with Y = (x_hat - x_t) / (1 - t). On the grid of the other samplers (t_i = i / N); no CFG. The control is EulerSamplerJiT on the head's mean
(the model's default output)."""
import torch

from src.diffusion.flow_matching.sampling_gmflow import _GMSampler


class DTMDrawSampler(_GMSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.order == 1, "DTMDrawSampler is first order only"

    def _impl_sampling(self, net, noise, condition, uncondition):
        steps = self._start(noise)
        dev = noise.device.type
        x = noise.float()
        x_trajs, v_trajs = [x], []
        for t_cur, t_next in zip(steps[:-1], steps[1:]):
            x0 = net(x, t_cur.repeat(x.shape[0]), condition, dtm_mode="draw").float()
            with torch.autocast(device_type=dev, enabled=False):
                sigma, sigma_to = (1 - t_cur).view(1, 1, 1, 1), (1 - t_next).view(1, 1, 1, 1)
                v = (x0 - x) / sigma
                x = x0 if float(sigma_to) == 0.0 else x + (sigma - sigma_to) * v
            v_trajs.append(v)
            x_trajs.append(x)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs
