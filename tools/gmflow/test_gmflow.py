#!/usr/bin/env python3
"""CPU tests of the GMFlow head (docs/gmflow.md): JiT_T2I(gm_k > 0), GMFlowTrainer (training_gmflow.py), GMSDESampler / GMODESampler (sampling_gmflow.py).
T11-T14 (docs/gmflow.md section 7): the per-pixel spread (gm_spread "pixel") and the second-order samplers (order 2).
Run on a vis node from baselines/PixelGen: CUDA_VISIBLE_DEVICES= PYTHONPATH=. python ../../tools/gmflow/test_gmflow.py [T1 T2 ...]
Each test prints PASS or FAIL; the exit code is the number of failed tests. tools/gmflow/mutate.py runs this file on mutated copies of the source."""
import math
import os
import subprocess
import sys
import traceback
import types

import torch
from torch.utils.flop_counter import FlopCounterMode

from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching import gmflow as G
from src.diffusion.flow_matching.sampling import EulerSamplerJiT
from src.diffusion.flow_matching.sampling_gmflow import GMODESampler, GMSDESampler
from src.diffusion.flow_matching.sampling_implicit import ImplicitSampler
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.flow_matching.training_gmflow import GMFlowTrainer
from src.diffusion.flow_matching.training_JiT import JiTTrainer, constant
from src.models.transformer.JiT_T2I import JiT_T2I

torch.set_default_dtype(torch.float32)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tools/gmflow/ -> repo root
BASE_COMMIT = "9711ff9"   # the last commit before the GMFlow code
SPREAD_BASE = "c16d71d"   # the last commit before the per-pixel spread and the second-order samplers
small = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=64, num_blocks=3, num_text_blocks=1, patch_size=16, txt_embed_dim=32, txt_max_length=32, bottleneck_dim=16)
d640 = dict(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256)
RESULTS = {}


def gm_trainer(**kw):
    args = dict(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, P_mean=-0.8, P_std=0.8, t_eps=0.05, null_condition_p=0.0, lam=0.5)
    args.update(kw)
    return GMFlowTrainer(**args)


def sampler(cls, n, **kw):
    return cls(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), guidance_fn=simple_guidance_fn, num_steps=n, guidance=1.0, **kw)


class FixedGM(torch.nn.Module):
    """Returns a fixed mixture and records the t it was called with."""
    def __init__(self, means, logweights, logstd):
        super().__init__(); self.means, self.logweights, self.logstd, self.t = means, logweights, logstd, None
    def forward(self, x_t, t, y, y_mask=None, gm_mode="full"):
        self.t = t.detach().clone()
        B = x_t.shape[0]
        return dict(means=self.means, logweights=self.logweights, logstd=self.logstd.expand(B, 1, 1, 1, 1))


class FixedPoint(torch.nn.Module):
    def __init__(self, x): super().__init__(); self.x, self.t = x, None
    def forward(self, x_t, t, y, y_mask=None):
        self.t = t.detach().clone()
        return self.x


def test(fn):
    RESULTS[fn.__name__] = fn
    return fn


@test
def T1():
    """K = 1, s_u fixed (s_x = c (1 - t)), lambda = 1: the gradient on mu_x is the baseline v-loss gradient times one constant, C / c^2, where 1 - t >= t_eps."""
    B, C, H, W = 64, 3, 8, 8
    torch.manual_seed(0)
    x = torch.randn(B, C, H, W); mu0 = torch.randn(B, C, H, W); c = 0.7
    mu_g = mu0.clone().requires_grad_(True)
    net_g = FixedGM(mu_g[:, None], torch.zeros(B, 1, 1, H, W), torch.full((1, 1, 1, 1, 1), math.log(c)))
    torch.manual_seed(1); gm_trainer(lam=1.0)._impl_trainstep(net_g, None, None, x, None, None)["loss"].backward()
    mu_b = mu0.clone().requires_grad_(True)
    net_b = FixedPoint(mu_b)
    jit = JiTTrainer(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, P_mean=-0.8, P_std=0.8, t_eps=0.05, null_condition_p=0.0)
    torch.manual_seed(1); jit._impl_trainstep(net_b, None, None, x, None, None)["loss"].backward()
    assert torch.allclose(net_g.t, net_b.t, atol=1e-6), "the two trainers drew different t"
    sigma = 1 - net_b.t
    outside = sigma >= 0.05
    assert outside.sum() >= 50, outside.sum()
    o = outside.view(B)
    gg, gb = mu_g.grad[o], mu_b.grad[o]
    err = ((gg - C / c ** 2 * gb).abs().max() / gb.abs().max()).item()   # whole tensor: an element ratio is fp32 noise where mu ~ x
    fit = ((gg * gb).sum() / (gb * gb).sum()).item()
    assert err < 1e-4 and abs(fit / (C / c ** 2) - 1) < 1e-5, (err, fit, C / c ** 2)
    inside = (~o).nonzero().view(-1)
    if len(inside):   # inside the clamp the baseline weight is 1 / t_eps^2, not 1 / sigma^2, so the ratio differs there
        k = inside[0]
        print(f"  inside the clamp (sigma {sigma[k].item():.4f}): ratio {(mu_g.grad[k] / mu_b.grad[k]).median().item():.4f}")
    print(f"  least-squares ratio of the gradients on mu outside the clamp: {fit:.6f} (expected C / c^2 = {C / c ** 2:.6f}); largest deviation {err:.1e} of the largest gradient; {int(o.sum())} samples, sigma in [{sigma[o].min().item():.3f}, {sigma[o].max().item():.3f}]")


@test
def T2():
    """The NLL written over x and written over v (u = (x_t - x) / sigma) differ by C log sigma per pixel, a constant in theta, so their gradients are equal."""
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    B, K, C, H, W = 4, 3, 3, 5, 5
    sigma = torch.rand(B, 1, 1, 1) * 0.9 + 0.05
    x = torch.randn(B, C, H, W); x_t = (1 - sigma) * x + sigma * torch.randn(B, C, H, W)
    mu = torch.randn(B, K, C, H, W, requires_grad=True); logits = torch.randn(B, K, 1, H, W, requires_grad=True); lsu = torch.randn(B, 1, 1, 1, 1, requires_grad=True)
    def nll_x():
        return G.gm_nll(x, mu, logits.log_softmax(1), lsu + torch.log(sigma[:, None]))
    def nll_u():
        u = (x_t - x) / sigma
        mu_u = (x_t[:, None] - mu) / sigma[:, None]
        return G.gm_nll(u, mu_u, logits.log_softmax(1), lsu)
    a, b = nll_x(), nll_u()
    diff = a - b
    assert torch.allclose(diff, (C * torch.log(sigma)).view(B, 1, 1).expand_as(diff), atol=1e-4), (diff - C * torch.log(sigma).view(B, 1, 1)).abs().max()
    ga = torch.autograd.grad(a.sum(), (mu, logits, lsu))
    gb = torch.autograd.grad(nll_u().sum(), (mu, logits, lsu))
    for p, q in zip(ga, gb):
        assert torch.allclose(p, q, rtol=1e-4, atol=1e-5), (p - q).abs().max()
    torch.set_default_dtype(torch.float32)
    print(f"  NLL_x - NLL_u = C log sigma to {(diff - C * torch.log(sigma).view(B, 1, 1)).abs().max().item():.2e}; gradients equal to {max((p - q).abs().max().item() for p, q in zip(ga, gb)):.2e}")


@test
def T3():
    """c1, c2, c3 in our convention: a Monte Carlo of the two-stage forward process with a fixed x0 gives the bridge's mean and variance; lambda = 1 gives c1 = 0, c2 = 1,
    c3 = 0, and the trainer's loss is then 2 x the plain per-pixel NLL of x under the x-mixture."""
    torch.manual_seed(0)
    N = 2_000_000
    for sh, lam in ((0.9, 0.5), (0.5, 0.5), (0.3, 0.25), (0.97, 0.8)):
        sl = sh * (1 - lam)
        x0 = 0.37
        xl = (1 - sl) * x0 + sl * torch.randn(N, dtype=torch.float64)
        xh = G.forward_transition(xl, torch.tensor(sl, dtype=torch.float64), torch.tensor(sh, dtype=torch.float64), torch.randn(N, dtype=torch.float64))
        assert abs(xh.mean().item() - (1 - sh) * x0) < 4e-3 and abs(xh.var().item() - sh ** 2) < 4e-3 * sh ** 2 * 3, (xh.mean().item(), xh.var().item())   # the marginal of x_high
        A = torch.stack([xh, torch.ones_like(xh)], 1)
        sol = torch.linalg.lstsq(A, xl[:, None]).solution.view(-1)
        resid = (xl - A @ sol).var().item()
        c1, c2, c3 = (v.item() for v in G.bridge_coefs(torch.tensor(sh, dtype=torch.float64), torch.tensor(sl, dtype=torch.float64)))
        assert abs(sol[0].item() - c1) < 3e-3 and abs(sol[1].item() - c2 * x0) < 3e-3 and abs(resid - c3) < 3e-3 * max(c3, 1e-3) * 10, (sh, lam, sol.tolist(), resid, c1, c2 * x0, c3)
        print(f"  sigma {sh} -> {sl:.3f}: MC slope {sol[0].item():.4f} vs c1 {c1:.4f}; intercept {sol[1].item():.4f} vs c2 x0 {c2 * x0:.4f}; residual var {resid:.5f} vs c3 {c3:.5f}")
    c1, c2, c3 = G.bridge_coefs(torch.tensor(0.6), torch.tensor(0.0))
    assert (c1.item(), c2.item(), c3.item()) == (0.0, 1.0, 0.0), (c1, c2, c3)
    # the trainer at lambda = 1 against the plain NLL
    B, K, C, H, W = 6, 4, 3, 4, 4
    x = torch.randn(B, C, H, W)
    mu, lw, ls = torch.randn(B, K, C, H, W), torch.randn(B, K, 1, H, W).log_softmax(1), torch.randn(1, 1, 1, 1, 1) * 0.3
    net = FixedGM(mu, lw, ls)
    loss = gm_trainer(lam=1.0)._impl_trainstep(net, None, None, x, None, None)["loss"]
    sigma = (1 - net.t).view(B, 1, 1, 1, 1)
    ref = 2 * G.gm_nll(x, mu, lw, ls + torch.log(sigma)).mean()
    assert torch.allclose(loss, ref, rtol=1e-5), (loss.item(), ref.item())
    print(f"  lambda = 1: c = (0, 1, 0); trainer loss {loss.item():.6f} = 2 x plain NLL {ref.item():.6f}")


def toy_posterior(c_high, sh, means, weights, s0):
    """The exact p(x0 | x_high = c_high) when x0 ~ sum_k w_k N(m_k, s0^2 I) and x_high = (1 - sh) x0 + sh eps: (posterior means (K, C), weights (K,), std)."""
    a = 1 - sh
    v = 1 / (1 / s0 ** 2 + a ** 2 / sh ** 2)
    pm = v * (means / s0 ** 2 + a * c_high / sh ** 2)
    ev_var = a ** 2 * s0 ** 2 + sh ** 2
    logw = torch.log(weights) - ((c_high - a * means) ** 2).sum(1) / (2 * ev_var)
    return pm, logw.softmax(0), math.sqrt(v)


@test
def T4():
    """The transition loss (lambda = 0.5) is minimized at the truth: for a fixed x_high with a known two-mode posterior, Adam on (mu_k, a_k, log s_u) from a random init
    recovers the posterior's modes, weights and std."""
    torch.manual_seed(0)
    C, sh, lam, s0 = 3, 0.6, 0.5, 0.15
    m = torch.tensor([[-0.5, 0.2, 0.8], [0.6, -0.4, 0.1]]); w = torch.tensor([0.3, 0.7])
    c_high = (1 - sh) * torch.tensor([0.1, -0.15, 0.4])
    pm, pw, ps = toy_posterior(c_high, sh, m, w, s0)
    assert 0.1 < pw.min() < 0.9, pw
    sl = sh * (1 - lam)
    c1, c2, c3 = G.bridge_coefs(torch.tensor(sh), torch.tensor(sl))
    mu = torch.randn(2, C, requires_grad=True); a = torch.randn(2, requires_grad=True); lsu = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([mu, a, lsu], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2500)
    sigh = torch.full((1, 1, 1, 1), sh); sigl = torch.full((1, 1, 1, 1), sl)
    for it in range(2500):
        n = 8192   # pixels of one image: one draw of x0 from the posterior each, then x_low from the bridge
        k = torch.multinomial(pw, n, replacement=True)
        x0 = pm[k] + ps * torch.randn(n, C)
        xl = c1 * c_high + c2 * x0 + c3.sqrt() * torch.randn(n, C)
        xl = xl.T.reshape(1, C, 1, n); xh = c_high.view(1, C, 1, 1).expand(1, C, 1, n)
        gm = dict(means=mu.view(1, 2, C, 1, 1).expand(1, 2, C, 1, n), logweights=a.log_softmax(0).view(1, 2, 1, 1, 1).expand(1, 2, 1, 1, n), logstd=lsu.view(1, 1, 1, 1, 1))
        loss = G.transition_nll(gm, xl, xh, sigh, sigl).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    order = mu[:, 0].argsort(); mu_s, w_s = mu.detach()[order], a.detach().softmax(0)[order]
    po = pm[:, 0].argsort(); pm_s, pw_s = pm[po], pw[po]
    s_x = sh * lsu.exp().item()
    print(f"  means {mu_s.numpy().round(3).tolist()} vs truth {pm_s.numpy().round(3).tolist()}; weights {w_s.numpy().round(3).tolist()} vs {pw_s.numpy().round(3).tolist()}; s_x {s_x:.4f} vs {ps:.4f}")
    assert (mu_s - pm_s).abs().max() < 0.03 and (w_s - pw_s).abs().max() < 0.03 and abs(s_x / ps - 1) < 0.08, "not recovered"


# two-point toy data: per pixel, P[0] with probability 0.3 and P[1] with 0.7 (3 channels)
P = torch.tensor([[1.0, -0.5, 0.3], [-0.8, 0.6, -0.2]]); PI = torch.tensor([0.3, 0.7])


def exact_posterior(x, t):
    """p(x0 | x_t) for the toy: weights over P, s = 0. x (B, 3, H, W), t (B,)."""
    B, C, H, W = x.shape
    a, s = t.view(B, 1, 1, 1, 1), (1 - t).view(B, 1, 1, 1, 1)
    means = P.view(1, 2, C, 1, 1).expand(B, 2, C, H, W)
    logw = torch.log(PI).view(1, 2, 1, 1, 1) - ((x[:, None] - a * means) ** 2).sum(2, keepdim=True) / (2 * s ** 2)
    return dict(means=means, logweights=logw.log_softmax(1), logstd=torch.full((B, 1, 1, 1, 1), -float("inf")))


def exact_net(x, t, c, gm_mode="mean"):
    gm = exact_posterior(x, t)
    return gm if gm_mode == "full" else G.gm_mean(gm)


def toy_stats(x):
    """Share of pixels within 0.05 of a data point, and the share at P[0]."""
    v = x.permute(0, 2, 3, 1).reshape(-1, 3)
    d = torch.cdist(v, P)
    near = d.min(1).values < 0.05
    return near.float().mean().item(), (d.argmin(1) == 0).float().mean().item(), d.min(1).values.max().item()


def marginal_err(x, t):
    """Largest relative error of the per-channel mean and variance of x against the exact marginal of x_t = t x0 + (1 - t) eps."""
    m0 = (PI[:, None] * P).sum(0); v0 = (PI[:, None] * P ** 2).sum(0) - m0 ** 2
    mean, var = t * m0, t ** 2 * v0 + (1 - t) ** 2
    em, ev = x.mean((0, 2, 3)), x.var((0, 2, 3))
    return max(((em - mean).abs() / var.sqrt()).max().item(), ((ev - var).abs() / var).max().item())


@test
def T5():
    """GM-SDE with the exact posterior reproduces the 2-point data at 4 and 50 calls (every pixel on a data point, P[0] at 0.3), and every intermediate state has the exact
    marginal; GM-ODE with the exact posterior lands on the data points; Euler on the posterior mean at 4 calls does not (control)."""
    torch.manual_seed(0)
    noise = torch.randn(64, 3, 32, 32)
    for n in (4, 50):
        xs, _ = sampler(GMSDESampler, n, rank_seed=False)._impl_sampling(exact_net, noise, None, None)
        near, f0, dmax = toy_stats(xs[-1])
        steps = sampler(GMSDESampler, n).timesteps
        merr = max(marginal_err(xs[i], steps[i].item()) for i in range(1, n))
        print(f"  GM-SDE {n} calls: on a data point {near:.4f} (max distance {dmax:.1e}), share of P[0] {f0:.4f}, largest marginal error {merr:.4f}")
        assert near == 1.0 and abs(f0 - 0.3) < 0.01 and merr < 0.03, (n, near, f0, merr)
    for n, sub in ((50, 3), (8, 16)):
        xs, _ = sampler(GMODESampler, n, substeps=sub, rank_seed=False)._impl_sampling(exact_net, noise, None, None)
        near, f0, dmax = toy_stats(xs[-1])
        print(f"  GM-ODE {n} calls x {sub} sub-steps: on a data point {near:.4f}, share of P[0] {f0:.4f}")
        assert near > 0.99 and abs(f0 - 0.3) < 0.02, (n, near, f0)
    xs, _ = sampler(EulerSamplerJiT, 4)._impl_sampling(lambda x, t, c: exact_net(x, t, c), noise, None, None)
    near, f0, _ = toy_stats(xs[-1])
    print(f"  control, Euler on the mean, 4 calls: on a data point {near:.4f}, share of P[0] {f0:.4f}")
    assert near < 0.99, near
    # the point-model path: K = 1, s = 0 is DDPM with the small variance; the last step returns the network's x
    point = JiT_T2I(**small)   # the real gm_k = 0 model accepts gm_mode and returns a tensor
    with torch.no_grad():
        assert torch.is_tensor(point(torch.randn(1, 3, 64, 64), torch.rand(1), torch.randn(1, 32, 32), gm_mode="full"))
    torch.manual_seed(5)
    xs, _ = sampler(GMSDESampler, 4, rank_seed=False)._impl_sampling(lambda x, t, c, gm_mode=None: exact_net(x, t, c), noise[:4], None, None)
    torch.manual_seed(5)
    x = noise[:4]
    steps = sampler(GMSDESampler, 4).timesteps
    for i in range(4):
        hat = exact_net(x, steps[i].repeat(4), None)
        s, s2 = 1 - steps[i], 1 - steps[i + 1]
        c1, c2, c3 = G.bridge_coefs(s, s2)
        x = c1 * x + c2 * hat + c3.sqrt() * torch.randn_like(x)
    assert torch.allclose(xs[-1], x, atol=1e-6), (xs[-1] - x).abs().max()
    print("  point model: GM-SDE = DDPM-small, step by step")


@test
def T6():
    """The GM-ODE sub-step: the closed-form re-weighted mixture (gm_convert) matches numerical Bayes on a grid, 1-D, K = 2."""
    torch.manual_seed(0)
    dt = torch.float64
    ss, st = torch.tensor(0.7, dtype=dt).view(1, 1, 1, 1), torch.tensor(0.45, dtype=dt).view(1, 1, 1, 1)
    means = torch.tensor([-0.6, 0.9], dtype=dt).view(1, 2, 1, 1, 1); lw = torch.tensor([0.35, 0.65], dtype=dt).log().view(1, 2, 1, 1, 1)
    s_u = 0.4; gm = dict(means=means, logweights=lw, logstd=torch.tensor(math.log(s_u), dtype=dt).view(1, 1, 1, 1, 1))
    xs, xt = torch.tensor(0.2, dtype=dt).view(1, 1, 1, 1), torch.tensor(-0.1, dtype=dt).view(1, 1, 1, 1)
    nm, nlw, nv = G.gm_convert(gm, ss, st, xs, xt)
    g = torch.linspace(-6, 6, 200001, dtype=dt)
    s_x = ss.item() * s_u
    q = sum(math.exp(lw.view(-1)[k].item()) * torch.exp(-(g - means.view(-1)[k]) ** 2 / (2 * s_x ** 2)) for k in range(2))
    ratio = torch.exp(-(xt.item() - (1 - st.item()) * g) ** 2 / (2 * st.item() ** 2) + (xs.item() - (1 - ss.item()) * g) ** 2 / (2 * ss.item() ** 2))
    post = q * ratio; post = post / torch.trapezoid(post, g)
    closed = sum(math.exp(nlw.view(-1)[k].item()) * torch.exp(-(g - nm.view(-1)[k]) ** 2 / (2 * nv.item())) / math.sqrt(2 * math.pi * nv.item()) for k in range(2))
    err = (post - closed).abs().max().item() / post.max().item()
    num_mean = torch.trapezoid(post * g, g).item()
    cm = G.gm_convert_mean(gm, ss, st, xs, xt).item()
    print(f"  grid vs closed form: largest density error {err:.2e} of the peak; mean {num_mean:.6f} vs {cm:.6f}")
    assert err < 1e-6 and abs(num_mean - cm) < 1e-6, (err, num_mean, cm)


@test
def T7():
    """Shapes and modes: a planted pattern comes back at its pixel, channel and component after unpatchify; the weights sum to 1 over K; the mean mode is sum A mu;
    the gm_k = 0 model is bit-identical to the model before this change (same parameters, same RNG stream, same outputs)."""
    torch.manual_seed(0)
    K, C, p = 4, 3, 16
    net = JiT_T2I(**small, gm_k=K)
    x = torch.randn(2, 3, 64, 64); t = torch.rand(2); y = torch.randn(2, 8, 32); mask = torch.ones(2, 8, dtype=torch.bool); mask[:, 5:] = False
    lin = net.final_layer.linear
    for (k, c, r, q, which) in ((2, 1, 5, 11, "mean"), (3, 0, 15, 0, "logit"), (0, 2, 0, 7, "mean")):
        lin.weight.data.zero_(); lin.bias.data.zero_()
        ch = k * C + c if which == "mean" else K * C + k
        lin.bias.data[ch * p * p + r * p + q] = 5.0
        with torch.no_grad():
            out = net(x, t, y, y_mask=mask, gm_mode="full")
        assert out["means"].shape == (2, K, C, 64, 64) and out["logweights"].shape == (2, K, 1, 64, 64) and out["logstd"].shape == (2, 1, 1, 1, 1)
        if which == "mean":
            want = torch.zeros(2, K, C, 64, 64); want[:, k, c, r::p, q::p] = 5.0
            assert torch.equal(out["means"], want), f"planted mean at k={k} c={c} ({r}, {q}) not found"
        else:
            lw = out["logweights"]
            hot = torch.zeros(2, K, 1, 64, 64, dtype=torch.bool); hot[:, k, 0, r::p, q::p] = True
            assert (lw[hot] > lw[~hot].max() - 1e-6).all() and torch.allclose(lw[:, k][~hot[:, k]], torch.full_like(lw[:, k][~hot[:, k]], -math.log(K))), "planted logit"
    lin.weight.data.normal_(std=0.05); lin.bias.data.normal_(std=0.05)
    with torch.no_grad():
        full, mean = net(x, t, y, y_mask=mask, gm_mode="full"), net(x, t, y, y_mask=mask)
    assert torch.allclose(full["logweights"].exp().sum(1), torch.ones(2, 1, 64, 64), atol=1e-5), "the weights do not sum to 1 over K"
    assert torch.allclose(mean, (full["logweights"].exp() * full["means"]).sum(1), atol=1e-6) and mean.shape == (2, 3, 64, 64)
    # off switch
    src = subprocess.run(["git", "-C", REPO, "show", f"{BASE_COMMIT}:baselines/PixelGen/src/models/transformer/JiT_T2I.py"], capture_output=True, text=True, check=True).stdout
    ref = types.ModuleType("JiT_T2I_ref"); exec(compile(src, "JiT_T2I_ref", "exec"), ref.__dict__)
    torch.manual_seed(7); a = ref.JiT_T2I(**small); after_a = torch.rand(3)
    torch.manual_seed(7); b = JiT_T2I(**small); after_b = torch.rand(3)
    assert torch.equal(after_a, after_b), "the off model draws another RNG stream at init"
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    for p_ in a.parameters():
        p_.data.normal_(std=0.05)
    b.load_state_dict(a.state_dict())
    y32 = torch.randn(2, 32, 32)
    with torch.no_grad():
        assert torch.equal(a(x, t, y32), b(x, t, y32)) and torch.equal(a(x, t, y, y_mask=mask), b(x, t, y, y_mask=mask)) and a(x, t, y, y_mask=mask).abs().max() > 1e-3
        with torch.autocast("cpu", dtype=torch.bfloat16):   # the training precision context
            assert torch.equal(a(x, t, y, y_mask=mask), b(x, t, y, y_mask=mask))
    print("  planted pattern found at its pixel, channel and component; weights sum to 1; mean mode = sum A mu; gm_k = 0 bit-identical to " + BASE_COMMIT)


@test
def T8():
    """Init: the K means differ; s_u = 1; the loss and the gradients are finite for t near 0 and near 1; the s-MLP's input is detached; the diagnostics appear only when asked."""
    torch.manual_seed(0)
    net = JiT_T2I(**small, gm_k=8)
    x = torch.randn(4, 3, 64, 64); y = torch.randn(4, 8, 32); mask = torch.ones(4, 8, dtype=torch.bool); mask[:, 6:] = False
    with torch.no_grad():
        out = net(x, torch.rand(4), y, y_mask=mask, gm_mode="full")
    spread = out["means"].std(1).mean().item()
    assert spread > 0.05 and torch.equal(out["logstd"], torch.zeros(4, 1, 1, 1, 1)), (spread, out["logstd"].view(-1))
    for P_mean in (-12.0, 12.0):   # t = sigmoid(P_mean): about 6e-6 and 1 - 6e-6
        net.zero_grad()
        tr = gm_trainer(P_mean=P_mean, P_std=0.0)
        tr.diag_now = True
        res = tr._impl_trainstep(net, None, None, x.clamp(-1, 1), y, {"y_mask": mask})
        res["loss"].backward()
        assert torch.isfinite(res["loss"]), res["loss"]
        bad = [k for k, p_ in net.named_parameters() if p_.grad is not None and not torch.isfinite(p_.grad).all()]
        assert not bad, bad
        assert "gm/nll" in res and torch.isfinite(res["gm/s_u"]) and torch.isfinite(res["gm/n_eff"]), res.keys()
        print(f"  t = sigmoid({P_mean}): loss {res['loss'].item():.4f}, n_eff {res['gm/n_eff'].item():.3f}, max weight {res['gm/max_w'].item():.3f}")
    assert net.t_embedder.mlp[0].weight.grad is not None
    # the s-MLP gets gradient, but none of it reaches the embedders through its input
    net.zero_grad()
    out = net(x, torch.rand(4), y, y_mask=mask, gm_mode="full")
    out["logstd"].sum().backward()
    assert net.gm_logstd[1].weight.grad.abs().max() > 0 or net.gm_logstd[3].weight.grad.abs().max() > 0
    assert all(p_.grad is None or p_.grad.abs().max() == 0 for p_ in net.t_embedder.parameters()) and all(p_.grad is None or p_.grad.abs().max() == 0 for p_ in net.y_embedder.parameters())
    res = gm_trainer()._impl_trainstep(net, None, None, x, y, {"y_mask": mask})
    assert set(res) == {"loss", "loss_per_sample"}, res.keys()
    print(f"  init: spread of the K means {spread:.3f}, s_u = 1; s-MLP input detached; no diagnostics unless asked")


@test
def T9():
    """FLOPs: FlopCounterMode on the d640 model (text length 8), the GMFlow head (training mode, gm_mode full) against the point model; the added FLOPs equal the hand count."""
    def fwd_flops(net, **kw):
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            net(torch.randn(1, 3, 256, 256), torch.rand(1), torch.randn(1, 8, 2048), y_mask=torch.ones(1, 8, dtype=torch.bool), **kw)
        return fc.get_total_flops()
    point = fwd_flops(JiT_T2I(**d640))
    REF, BASE_STEPS = 34.72e9, 66816
    for K in (8, 1):
        gm = fwd_flops(JiT_T2I(**d640, gm_k=K), gm_mode="full")
        mean_mode = fwd_flops(JiT_T2I(**d640, gm_k=K))
        for spread in ("image", "pixel"):
            gm = fwd_flops(JiT_T2I(**d640, gm_k=K, gm_spread=spread), gm_mode="full")
            hand = 2 * 256 * 640 * (K * 4 * 256 - 768) + 2 * (640 * 1024 + 1024 * 1) + (2 * 256 * 640 * 256 if spread == "pixel" else 0)
            assert gm - point == hand, (K, spread, gm - point, hand)
            r = (gm - point) / REF
            steps = int(BASE_STEPS / (1 + r)) // 128 * 128
            while (steps + 128) * (1 + r) <= BASE_STEPS:
                steps += 128
            assert steps * (1 + r) <= BASE_STEPS < (steps + 128) * (1 + r)
            if spread == "pixel":
                assert steps == {8: 62208, 1: 66432}[K], (K, steps)
            print(f"  K={K} spread {spread}: point {point / 1e9:.4f} GFLOP, GMFlow {gm / 1e9:.4f} (mean mode {mean_mode / 1e9:.4f}); added {(gm - point) / 1e6:.2f} MFLOP = r {r:.5f} of 34.72; STEPS {steps} (x (1 + r) = {steps * (1 + r):.0f} <= {BASE_STEPS})")


@test
def T10():
    """Extra: GM-ODE with one sub-step is the exact DDIM step on the mixture mean (ImplicitSampler on the mean mode); GM-SDE final_mean ends on the mixture mean;
    the samplers assert CFG 1.0; the trainer and samplers run on the real small model with the masked text path."""
    torch.manual_seed(0)
    net = JiT_T2I(**small, gm_k=3)
    for p_ in net.parameters():
        p_.data.add_(torch.randn_like(p_) * 0.02)
    y = torch.randn(2, 8, 32); mask = torch.ones(2, 8, dtype=torch.bool); mask[:, 4:] = False
    f = lambda x, t, c, **kw: net(x, t, c, y_mask=mask, **kw)
    noise = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        a = sampler(GMODESampler, 5, substeps=1, rank_seed=False)._impl_sampling(f, noise, y, None)[0][-1]
        b = sampler(ImplicitSampler, 5)._impl_sampling(f, noise, y, None)[0][-1]
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
        torch.manual_seed(3); xs, _ = sampler(GMSDESampler, 4, final_mean=True, rank_seed=False)._impl_sampling(f, noise, y, None)
        last = f(xs[-2], torch.full((2,), 0.75), y)
        assert torch.allclose(xs[-1], last, atol=1e-5), (xs[-1] - last).abs().max()
        for cls in (GMSDESampler, GMODESampler):
            s = cls(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), guidance_fn=simple_guidance_fn, num_steps=4, guidance=2.0)
            try:
                s._impl_sampling(f, noise, y, None); raise RuntimeError("no assert on CFG")
            except AssertionError:
                pass
        o = sampler(GMODESampler, 8, substeps=16, rank_seed=False)._impl_sampling(f, noise, y, None)[0][-1]
        assert torch.isfinite(o).all()
    print("  GM-ODE n = 1 = exact DDIM on the mean; final_mean ends on the mean; CFG asserted")



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# T11-T14 (docs/gmflow.md section 7): the per-pixel spread and the second-order samplers

def load_at(commit, path, name, drop=(), inject=None):
    """A module from a file at a commit (git show), with the import lines in drop removed and the names in inject put in first."""
    src = subprocess.run(["git", "-C", REPO, "show", f"{commit}:baselines/PixelGen/{path}"], capture_output=True, text=True, check=True).stdout
    for line in drop:
        assert src.count(line) == 1, line
        src = src.replace(line, "")
    mod = types.ModuleType(name)
    mod.__dict__.update(inject or {})
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


def model_io(K, B=2):
    torch.manual_seed(11)
    x = torch.randn(B, 3, 64, 64); t = torch.rand(B); y = torch.randn(B, 8, 32); mask = torch.ones(B, 8, dtype=torch.bool); mask[:, 5:] = False
    return x, t, y, mask


def full_loss(out, seed=3):
    g = torch.Generator().manual_seed(seed)
    return ((out["means"].float() * torch.randn(out["means"].shape, generator=g)).sum() + (out["logweights"] * torch.randn(out["logweights"].shape, generator=g)).sum()
            + 3 * out["logstd"].sum())


@test
def T11():
    """gm_spread "image" is bit-identical to c16d71d at K = 8 and K = 1 (parameters, RNG stream, outputs and gradients, fp32 and bf16 autocast); "pixel" at init equals
    "image" (same parameters and RNG stream, outputs equal, log s_u (B, 1, 1, H, W) constant over pixels); the offset head gets gradient and its input is detached."""
    ref = load_at(SPREAD_BASE, "src/models/transformer/JiT_T2I.py", "JiT_T2I_c16d71d")
    for K in (8, 1):
        x, t, y, mask = model_io(K)
        torch.manual_seed(7); a = ref.JiT_T2I(**small, gm_k=K); ra = torch.rand(3)
        torch.manual_seed(7); b = JiT_T2I(**small, gm_k=K); rb = torch.rand(3)
        torch.manual_seed(7); c = JiT_T2I(**small, gm_k=K, gm_spread="pixel"); rc = torch.rand(3)
        assert torch.equal(ra, rb) and torch.equal(ra, rc), "another RNG stream at init"
        sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
        assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
        assert set(sc) - set(sa) == {"gm_spread_head.weight", "gm_spread_head.bias"} and all(torch.equal(sa[k], sc[k]) for k in sa)
        assert c.gm_spread_head.weight.abs().max() == 0 and c.gm_spread_head.bias.abs().max() == 0
        # pixel at init = image
        for ac in (False, True):
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=ac):
                oa, oc = b(x, t, y, y_mask=mask, gm_mode="full"), c(x, t, y, y_mask=mask, gm_mode="full")
                assert torch.equal(b(x, t, y, y_mask=mask), c(x, t, y, y_mask=mask))
            assert oc["logstd"].shape == (2, 1, 1, 64, 64) and oa["logstd"].shape == (2, 1, 1, 1, 1), oc["logstd"].shape
            assert torch.equal(oa["means"], oc["means"]) and torch.equal(oa["logweights"], oc["logweights"]) and torch.equal(oa["logstd"].expand_as(oc["logstd"]), oc["logstd"])
        # image = c16d71d after training-like weights: outputs and gradients, fp32 and bf16
        for p_ in a.parameters():
            p_.data.normal_(std=0.05)
        b.load_state_dict(a.state_dict())
        for ac in (False, True):
            for net in (a, b):
                net.zero_grad()
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=ac):
                    out = net(x, t, y, y_mask=mask, gm_mode="full")
                    mean = net(x, t, y, y_mask=mask)
                full_loss(out).backward()
                net._o = (out, mean)
            (oa, ma), (ob, mb) = a._o, b._o
            assert all(torch.equal(oa[k], ob[k]) for k in oa) and torch.equal(ma, mb), f"K={K} autocast={ac}: outputs differ"
            ga, gb = dict(a.named_parameters()), dict(b.named_parameters())
            bad = [k for k in ga if (ga[k].grad is None) != (gb[k].grad is None) or (ga[k].grad is not None and not torch.equal(ga[k].grad, gb[k].grad))]
            assert not bad, f"K={K} autocast={ac}: gradients differ at {bad[:4]}"
        # the offset head: gradient at init through the loss; with nonzero weights, no gradient reaches the trunk through its input
        c.zero_grad()
        tr = gm_trainer(); tr.diag_now = True
        res = tr._impl_trainstep(c, None, None, x.clamp(-1, 1), y, {"y_mask": mask})
        res["loss"].backward()
        assert c.gm_spread_head.weight.grad.abs().max() > 0 and c.gm_spread_head.bias.grad.abs().max() > 0, "the offset head gets no gradient"
        assert torch.isfinite(res["gm/logsu_pix_std"]) and res["gm/logsu_pix_std"] == 0, res["gm/logsu_pix_std"]
        for p_ in c.parameters():
            p_.data.add_(torch.randn_like(p_) * 0.05)
        c.zero_grad()
        out = c(x, t, y, y_mask=mask, gm_mode="full")
        assert out["logstd"].flatten(1).std(1).min() > 0, "the spread does not vary over pixels"
        out["logstd"].sum().backward()
        leak = [k for k, p_ in c.named_parameters() if not k.startswith(("gm_spread_head", "gm_logstd")) and p_.grad is not None and p_.grad.abs().max() > 0]
        assert not leak and c.gm_spread_head.weight.grad.abs().max() > 0, f"gradient through the offset head's input reaches {leak[:4]}"
        for P_mean in (-12.0, 12.0):
            c.zero_grad()
            res = gm_trainer(P_mean=P_mean, P_std=0.0)._impl_trainstep(c, None, None, x.clamp(-1, 1), y, {"y_mask": mask})
            res["loss"].backward()
            assert torch.isfinite(res["loss"]) and all(torch.isfinite(p_.grad).all() for p_ in c.parameters() if p_.grad is not None), (K, P_mean)
        print(f"  K={K}: image = {SPREAD_BASE} bitwise (outputs and gradients, fp32 and bf16); pixel at init = image; offset head trained, input detached; "
              f"log s_u over pixels after a perturbation: std {out['logstd'].flatten(1).std(1).mean().item():.4f}")


@test
def T12():
    """A known posterior whose std differs between pixels (4 groups of pixels): Adam on the transition loss (lambda 1 and 0.5) over the shared means and weights and one
    log s_u per group, given to the loss as a per-pixel logstd (1, 1, 1, 1, n), recovers the four stds, the means and the weights."""
    C, sh, n = 3, 0.6, 32768
    m = torch.tensor([[-0.5, 0.2, 0.8], [0.6, -0.4, 0.1]]); w = torch.tensor([0.3, 0.7])
    stds = torch.tensor([0.10, 0.17, 0.25, 0.35])
    group = torch.arange(n) % 4
    c_high = (1 - sh) * torch.tensor([0.1, -0.15, 0.4])
    for lam, iters in ((1.0, 2500), (0.5, 6000)):   # at lambda 0.5, c3 = 0.083 is 20 x c2^2 s_x^2 at s_x = 0.1: the loss is flat in s_x there (0.1 and 0.113 differ by
        torch.manual_seed(0)                       # 1e-4), and 2500 steps left the smallest std 13% off (seeds 1, 2: 0.1% and 0.7%)
        sl = sh * (1 - lam)
        c1, c2, c3 = G.bridge_coefs(torch.tensor(sh), torch.tensor(sl))
        mu = torch.randn(2, C, requires_grad=True); a = torch.randn(2, requires_grad=True); lsu = torch.zeros(4, requires_grad=True)
        opt = torch.optim.Adam([mu, a, lsu], lr=0.02)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        sigh = torch.full((1, 1, 1, 1), sh); sigl = torch.full((1, 1, 1, 1), sl)
        for it in range(iters):
            k = torch.multinomial(w, n, replacement=True)
            x0 = m[k] + stds[group][:, None] * torch.randn(n, C)
            xl = c1 * c_high + c2 * x0 + c3.clamp(min=0).sqrt() * torch.randn(n, C)
            xl = xl.T.reshape(1, C, 1, n); xh = c_high.view(1, C, 1, 1).expand(1, C, 1, n)
            gm = dict(means=mu.view(1, 2, C, 1, 1).expand(1, 2, C, 1, n), logweights=a.log_softmax(0).view(1, 2, 1, 1, 1).expand(1, 2, 1, 1, n),
                      logstd=lsu[group].view(1, 1, 1, 1, n))
            loss = G.transition_nll(gm, xl, xh, sigh, sigl).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        order = mu[:, 0].argsort(); mu_s, w_s = mu.detach()[order], a.detach().softmax(0)[order]
        po = m[:, 0].argsort()
        s_x = sh * lsu.detach().exp()
        rel = (s_x / stds - 1).abs()
        print(f"  lambda {lam}, {iters} steps: s_x per group {s_x.numpy().round(4).tolist()} vs {stds.numpy().round(4).tolist()} (largest relative error {rel.max().item():.3f}); means error "
              f"{(mu_s - m[po]).abs().max().item():.4f}; weights {w_s.numpy().round(3).tolist()}")
        assert rel.max() < 0.05 and (mu_s - m[po]).abs().max() < 0.03 and (w_s - w[po]).abs().max() < 0.03, ("not recovered", lam)


def grid_mixture(g, means, lw, var):
    """1-D mixture density on the grid g: means, lw (K,), var scalar."""
    return sum(math.exp(lw[k]) * torch.exp(-(g - means[k]) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var) for k in range(len(means)))


def exact_posterior_px(x, t, s0):
    """p(x0 | x_t) when, at each pixel, x0 ~ sum_k PI_k N(P_k, s0^2 I) with s0 (1, 1, 1, H, W) per pixel (0 = the two-point data): K = 2, logstd (B, 1, 1, H, W)
    in velocity units (s_u = sqrt(posterior var) / sigma)."""
    B, C, H, W = x.shape
    a, s = t.view(B, 1, 1, 1, 1), (1 - t).view(B, 1, 1, 1, 1)
    P5 = P.view(1, 2, C, 1, 1)
    den = s ** 2 + a ** 2 * s0 ** 2                                                        # (B, 1, 1, H, W)
    means = (s ** 2 * P5 + a * s0 ** 2 * x[:, None]) / den
    logw = torch.log(PI).view(1, 2, 1, 1, 1) - ((x[:, None] - a * P5) ** 2).sum(2, keepdim=True) / (2 * den)
    return dict(means=means, logweights=logw.log_softmax(1), logstd=torch.log(s0) - 0.5 * torch.log(den))


def px_net(s0):
    def f(x, t, c, gm_mode="mean"):
        gm = exact_posterior_px(x, t, s0)
        return gm if gm_mode == "full" else G.gm_mean(gm)
    return f


S0_GROUPS = (0.0, 0.02, 0.05)


def s0_map(W=32, groups=S0_GROUPS):
    return torch.tensor([groups[j % len(groups)] for j in range(W)]).view(1, 1, 1, 1, W)


def landing(x, s0):
    """Errors of the final samples against the per-pixel data: |share of P[0] - 0.3|; per s0 group, the largest distance to a data point (s0 = 0) or |std of the
    residual per channel / s0 - 1| (s0 > 0); the share of s0 = 0 pixels within 0.05 of a data point."""
    B, C, H, W = x.shape
    v = x.permute(0, 2, 3, 1).reshape(-1, C)
    sv = s0.view(1, 1, W).expand(B, H, W).reshape(-1)
    d = torch.cdist(v, P)
    near = d.argmin(1)
    out = {"f0": abs((near == 0).float().mean().item() - 0.3)}
    for g in sorted(set(sv.tolist())):
        sel = sv == g
        if g == 0:
            out["dist0"] = d.min(1).values[sel].max().item()
            out["near0"] = (d.min(1).values[sel] < 0.05).float().mean().item()
        else:
            resid = v[sel] - P[near[sel]]
            out[f"std{g:g}"] = abs(resid.std().item() / g - 1)
    return out


@test
def T13():
    """Per-pixel spread in the samplers and the loss: GM-ODE's sub-step (gm_convert) and the transition loss match numerical Bayes / integration on a grid at every
    pixel; GM-SDE's draw has each pixel's mean and variance; with the exact posterior of per-pixel data (s0 = 0, 0.02, 0.05 by column) as the network, GM-SDE (4, 50
    calls) and GM-ODE (8 x 16, 50 x 3) land every s0 = 0 pixel on a data point and give the other pixels their own spread."""
    torch.manual_seed(0)
    dt = torch.float64
    Wp = 4
    means = torch.randn(1, 2, 1, 1, Wp, dtype=dt); lw = torch.randn(1, 2, 1, 1, Wp, dtype=dt).log_softmax(1)
    s_u = torch.tensor([0.2, 0.4, 0.7, 1.1], dtype=dt)
    gm = dict(means=means, logweights=lw, logstd=s_u.log().view(1, 1, 1, 1, Wp))
    ss, st = torch.tensor(0.7, dtype=dt).view(1, 1, 1, 1), torch.tensor(0.45, dtype=dt).view(1, 1, 1, 1)
    xs, xt = torch.randn(1, 1, 1, Wp, dtype=dt) * 0.5, torch.randn(1, 1, 1, Wp, dtype=dt) * 0.5
    nm, nlw, nv = G.gm_convert(gm, ss, st, xs, xt)
    assert nv.shape == (1, 1, 1, 1, Wp), nv.shape
    cm = G.gm_convert_mean(gm, ss, st, xs, xt)
    g = torch.linspace(-8, 8, 200001, dtype=dt)
    errs = []
    for j in range(Wp):
        s_x2 = (ss.item() * s_u[j].item()) ** 2
        q = grid_mixture(g, means[0, :, 0, 0, j].tolist(), lw[0, :, 0, 0, j].tolist(), s_x2)
        ratio = torch.exp(-(xt[0, 0, 0, j].item() - (1 - st.item()) * g) ** 2 / (2 * st.item() ** 2) + (xs[0, 0, 0, j].item() - (1 - ss.item()) * g) ** 2 / (2 * ss.item() ** 2))
        post = q * ratio; post = post / torch.trapezoid(post, g)
        closed = grid_mixture(g, nm[0, :, 0, 0, j].tolist(), nlw[0, :, 0, 0, j].tolist(), nv[0, 0, 0, 0, j].item())
        errs.append(max((post - closed).abs().max().item() / post.max().item(), abs(torch.trapezoid(post * g, g).item() - cm[0, 0, 0, j].item())))
    print(f"  GM-ODE sub-step per pixel: largest error against grid Bayes over {Wp} pixels with s_u {s_u.tolist()}: {max(errs):.2e}")
    assert max(errs) < 1e-6, errs
    # the transition loss per pixel against the integral over x0 on the grid
    sh, sl = torch.tensor(0.8, dtype=dt).view(1, 1, 1, 1), torch.tensor(0.4, dtype=dt).view(1, 1, 1, 1)
    xh, xl = torch.randn(1, 1, 1, Wp, dtype=dt) * 0.5, torch.randn(1, 1, 1, Wp, dtype=dt) * 0.5
    nll = G.transition_nll(gm, xl, xh, sh, sl)
    c1, c2, c3 = (v.item() for v in G.bridge_coefs(sh, sl))
    terr = []
    for j in range(Wp):
        q = grid_mixture(g, means[0, :, 0, 0, j].tolist(), lw[0, :, 0, 0, j].tolist(), (sh.item() * s_u[j].item()) ** 2)
        lik = torch.exp(-(xl[0, 0, 0, j].item() - c1 * xh[0, 0, 0, j].item() - c2 * g) ** 2 / (2 * c3)) / math.sqrt(2 * math.pi * c3)
        ref = -math.log(torch.trapezoid(q * lik, g).item()) - 0.5 * math.log(2 * math.pi)   # gm_nll drops C/2 log 2 pi
        terr.append(abs(ref - nll[0, 0, j].item()))
    print(f"  transition loss per pixel against the grid integral: largest error {max(terr):.2e}")
    assert max(terr) < 1e-6, terr
    # GM-SDE's draw per pixel
    Bd = 200000
    gmd = dict(means=means.float().expand(Bd, -1, 3, -1, -1).contiguous(), logweights=lw.float().expand(Bd, -1, -1, -1, -1), logstd=s_u.float().log().view(1, 1, 1, 1, Wp).expand(Bd, -1, -1, -1, -1))
    sig = torch.full((Bd, 1, 1, 1), 0.6)
    d = G.gm_draw(gmd, sig)                                                                 # (Bd, 3, 1, Wp)
    A = lw.exp()[0, :, 0, 0, :]; mu = means[0, :, 0, 0, :]
    want_m = (A * mu).sum(0); want_v = (A * mu ** 2).sum(0) - want_m ** 2 + (0.6 * s_u) ** 2
    em, ev = d.double().mean((0, 1, 2)), d.double().var((0, 2)).mean(0)
    derr = max(((em - want_m).abs() / want_v.sqrt()).max().item(), ((ev / want_v) - 1).abs().max().item())
    print(f"  GM-SDE draw per pixel: largest error of mean / std and variance ratio {derr:.4f}")
    assert derr < 0.01, derr
    # landing with per-pixel data
    s0 = s0_map()
    noise = torch.randn(64, 3, 32, 32)
    for name, cls, n, kw in (("GM-SDE", GMSDESampler, 4, {}), ("GM-SDE", GMSDESampler, 50, {}), ("GM-ODE", GMODESampler, 8, {"substeps": 16}), ("GM-ODE", GMODESampler, 50, {"substeps": 3})):
        torch.manual_seed(1)
        xs_, _ = sampler(cls, n, rank_seed=False, **kw)._impl_sampling(px_net(s0), noise, None, None)
        e = landing(xs_[-1], s0)
        print(f"  {name} {n} calls: " + ", ".join(f"{k} {v:.4f}" for k, v in e.items()))
        tol = 0.05 if cls is GMSDESampler else 0.2   # GM-ODE is deterministic: its discretization error shows in a small spread (8 x 16: +14% at s0 = 0.02)
        assert e["near0"] == 1.0 and e["f0"] < 0.015 and all(v < tol for k, v in e.items() if k.startswith("std")), (name, n, e)


def ref_iso(means, logstds, lw):
    """GMFlow gm_to_iso_gaussian_jit + the no-guidance branch of gm_2nd_order (var averaged over H, W), transcribed. x-space logstds."""
    wts = lw.exp()
    g_mean = (wts * means).sum(-4, keepdim=True)
    diffs = means - g_mean
    g_var = (wts * (diffs * diffs)).sum(-4, keepdim=True).mean(-3, keepdim=True) + (logstds * 2).exp()
    return g_mean.squeeze(-4), g_var.squeeze(-4).mean(dim=(-1, -2), keepdim=True)


def ref_convert_to_mean(sigma_src, sigma_tgt, x_t_src, x_t_tgt, gm_means, gm_vars, gm_logweights, eps=1e-6):
    """GMFlow denoising_gm_convert_to_mean_jit, transcribed."""
    alpha_src, alpha_tgt = 1 - sigma_src, 1 - sigma_tgt
    a_t_s, a_s_t = alpha_tgt * sigma_src, alpha_src * sigma_tgt
    denom = (a_t_s.square() - a_s_t.square()).clamp(min=eps)
    g_mean = ((a_t_s * sigma_src * x_t_tgt - a_s_t * sigma_tgt * x_t_src) / denom).unsqueeze(-4)
    g_var = ((sigma_tgt * sigma_src).square() / denom).unsqueeze(-4)
    norm = (g_var + gm_vars).clamp(min=eps)
    out_means = (g_var * gm_means + gm_vars * g_mean) / norm
    out_w = (gm_logweights + (gm_means - g_mean).square().sum(dim=-3, keepdim=True) * (-0.5 / norm)).softmax(dim=-4)
    return (out_means * out_w).sum(dim=-4)


def ref_gm_2nd_order(gm_x, x_t, t, h, prev_gm_x, prev_x_t, prev_t, prev_h, ca=0.005, num_timesteps=1000):
    """GMFlow GMFlowMixin.gm_2nd_order, branch prev_gm is not None and cfg_bias is None, transcribed (gmflow.py:251-291, gmflow_ops gm_mul_iso_gaussian and
    iso_gaussian_mul_iso_gaussian). t, h, prev_t, prev_h in the code's timesteps (sigma x 1000); gm_x and prev_gm_x in x-space with 'logstds'."""
    gm_mean, base_var = ref_iso(gm_x["means"], gm_x["logstds"], gm_x["logweights"])
    avg_var = base_var.mean(dim=(-3, -2, -1), keepdim=True)
    mean_from_prev = ref_convert_to_mean(prev_t / num_timesteps, t / num_timesteps, prev_x_t, x_t, prev_gm_x["means"], (prev_gm_x["logstds"] * 2).exp(), prev_gm_x["logweights"])
    k = 0.5 * h / prev_h
    prev_h_norm = prev_h / num_timesteps
    err_power = avg_var * (0.0 * 0.0 + ca)
    mean_diff = (gm_mean - mean_from_prev) * ((1 - err_power / (prev_h_norm * prev_h_norm)).clamp(min=0).sqrt() * k)
    bias = mean_diff
    bias_power = bias.square().mean(dim=(-3, -2, -1), keepdim=True)
    bias = bias * (avg_var / bias_power.clamp(min=1e-6)).clamp(max=1).sqrt()
    out_mean, out_var = gm_mean + bias, base_var * (1 - bias_power / avg_var.clamp(min=1e-6)).clamp(min=1e-6)
    norm = (1 * base_var + (-1) * out_var).clamp(min=1e-6)                            # iso_gaussian_mul_iso_gaussian(out, base, 1, -1)
    r_var, r_mean = base_var * out_var / norm, (1 * base_var * out_mean + (-1) * out_var * gm_mean) / norm
    g_mean, g_var, g_logstd = r_mean.unsqueeze(-4), r_var.unsqueeze(-4), (torch.log(r_var) / 2).unsqueeze(-4)
    gm_vars = (gm_x["logstds"] * 2).exp()
    norm2 = (g_var + 1.0 * gm_vars).clamp(min=1e-6)                                    # gm_mul_iso_gaussian(gm, ratio, 1, 1)
    out = dict(means=(g_var * gm_x["means"] + gm_vars * g_mean) / norm2,
               logstds=gm_x["logstds"] + g_logstd - 0.5 * torch.log(norm2),
               logweights=(gm_x["logweights"] + (gm_x["means"] - g_mean).square().sum(dim=-3, keepdim=True) * (-0.5 / norm2)).log_softmax(dim=-4))
    return out, dict(k=k, err_power=err_power, mean_diff=mean_diff, bias_power=bias_power, bias=bias, avg_var=avg_var, new_var=out_var, mean_prev=mean_from_prev)


def rand_gm(B, K, C, H, W, pixel, dt=torch.float64, scale=1.0, s=0.3):
    ls = (torch.randn(B, 1, 1, H, W, dtype=dt) * 0.3 if pixel else torch.randn(B, 1, 1, 1, 1, dtype=dt) * 0.3) + math.log(s)
    return dict(means=torch.randn(B, K, C, H, W, dtype=dt) * scale, logweights=torch.randn(B, K, 1, H, W, dtype=dt).log_softmax(1), logstd=ls)


def mixture_err(a, b):
    """Largest difference between two mixtures (velocity-unit logstd): means and weights where the weight matters, and log s."""
    wa, wb = a["logweights"].exp(), b["logweights"].exp()
    return max((wa - wb).abs().max().item(), ((a["means"] - b["means"]).abs() * wa).max().item(), (a["logstd"] - b["logstd"]).abs().max().item())


@test
def T14():
    """Second order (GM-SDE 2, GM-ODE 2): its quantities (k, err_power, the clamp at 0, the bias cap, the new variance) and its output equal a transcription of the
    official code on random inputs (image and per-pixel spread); the reweighted mixture equals grid Bayes per pixel; zero correction when the previous mixture converted
    to x_t equals the current one; the first call is first order; order 1 is bit-identical to c16d71d; with the exact posterior of the two-point and the per-pixel
    data, GM-SDE 2 and GM-ODE 2 land on the data at 4 and 8 calls with errors no larger than first order."""
    torch.manual_seed(0)
    dt = torch.float64
    B, K, C, H, W = 3, 4, 3, 8, 8
    s2 = sampler(GMODESampler, 8, order=2)
    sig = torch.tensor(0.55, dtype=dt).view(1, 1, 1, 1)
    worst = {}
    for pixel in (False, True):
        for regime, (sig_prev, sig_to, shift) in {"plain": (0.7, 0.4, 0.03), "clamp at 0": (0.5503, 0.4, 0.03), "bias cap": (0.7, 0.4, 3.0)}.items():
            gm, gp = rand_gm(B, K, C, H, W, pixel), rand_gm(B, K, C, H, W, pixel)
            x, xp = torch.randn(B, C, H, W, dtype=dt), torch.randn(B, C, H, W, dtype=dt)
            sp, st_ = torch.tensor(sig_prev, dtype=dt).view(1, 1, 1, 1), torch.tensor(sig_to, dtype=dt).view(1, 1, 1, 1)
            mean_prev_ours = G.gm_convert_mean(gp, sp.expand(B, -1, -1, -1), sig.expand(B, -1, -1, -1), xp, x)
            # shift the current means so that mean - mean_prev has a chosen size
            mean_cur = G.gm_mean(gm)
            gm["means"] = gm["means"] - mean_cur.unsqueeze(1) + mean_prev_ours.unsqueeze(1) + shift * torch.randn(B, 1, C, H, W, dtype=dt)
            prev = (gp, xp, sp, (sp - sig).item())
            out, prev_new = s2._second_order(gm, x, sig, st_, prev)
            h, ph = (sig - st_).item(), (sp - sig).item()
            _, q = G.gm_second_order(gm, mean_prev_ours, sig.view(1).expand(B), h, ph, parts=True)
            to_x = lambda g_, s_: dict(means=g_["means"], logweights=g_["logweights"], logstds=g_["logstd"] + torch.log(s_.view(1, 1, 1, 1, 1)))
            rout, rq = ref_gm_2nd_order(to_x(gm, sig), x, sig * 1000, h * 1000, to_x(gp, sp), xp, sp * 1000, ph * 1000)
            rout_u = dict(means=rout["means"], logweights=rout["logweights"], logstd=rout["logstds"] - torch.log(sig.view(1, 1, 1, 1, 1)))
            for name in ("err_power", "mean_diff", "bias_power", "bias", "avg_var", "new_var"):
                assert torch.allclose(q[name], rq[name], rtol=1e-9, atol=1e-12), (pixel, regime, name, (q[name] - rq[name]).abs().max())
            assert abs(q["k"] - rq["k"]) < 1e-12 and torch.allclose(mean_prev_ours, rq["mean_prev"], atol=1e-12), (pixel, regime)
            e = mixture_err(out, rout_u)
            worst[(pixel, regime)] = e
            if regime == "clamp at 0":
                assert (q["scale"] == 0).all() and (q["bias"] == 0).all() and mixture_err(out, gm) < 1e-12, "the clamp at 0 must leave the mixture unchanged"
                assert e < 1e-4, (pixel, regime, e)   # the code's clamp on var - v' moves the mixture by ~1e-6 s_x^2 / var^2
            elif regime == "bias cap":
                assert torch.allclose(q["bias"].square().mean((-3, -2, -1), keepdim=True), q["avg_var"], rtol=1e-9), "bias power not capped at avg_var"
                assert e < 1e-6, (pixel, regime, e)
            else:
                assert (q["bias_power"] < q["avg_var"]).all() and (q["scale"] > 0).all(), "the plain regime hit a clamp"
                assert e < 1e-9, (pixel, regime, e)
            assert prev_new[0] is gm, "the next call must convert the raw mixture"
    print("  against the transcribed official code, largest mixture difference: " + "; ".join(f"{'pixel' if p else 'image'} {r} {v:.1e}" for (p, r), v in worst.items()))
    # the reweighted mixture against grid Bayes per pixel (C = 1)
    Wp = 4
    gm = rand_gm(1, 3, 1, 1, Wp, True, scale=0.6)
    mean_prev = G.gm_mean(gm) + 0.05 * torch.randn(1, 1, 1, Wp, dtype=dt)
    out, q = G.gm_second_order(gm, mean_prev, sig.view(1), 0.15, 0.15, parts=True)
    g = torch.linspace(-8, 8, 200001, dtype=dt)
    V, Vn = q["var"].item(), q["new_var"].item()
    errs = []
    for j in range(Wp):
        s_x2 = math.exp(2 * (gm["logstd"][0, 0, 0, 0, j].item() + math.log(sig.item())))
        dens = grid_mixture(g, gm["means"][0, :, 0, 0, j].tolist(), gm["logweights"][0, :, 0, 0, j].tolist(), s_x2)
        m_, b_ = q["mean"][0, 0, 0, j].item(), q["bias"][0, 0, 0, j].item()
        post = dens * torch.exp(-(g - m_ - b_) ** 2 / (2 * Vn) + (g - m_) ** 2 / (2 * V)); post = post / torch.trapezoid(post, g)
        so2 = math.exp(2 * (out["logstd"][0, 0, 0, 0, j].item() + math.log(sig.item())))
        closed = grid_mixture(g, out["means"][0, :, 0, 0, j].tolist(), out["logweights"][0, :, 0, 0, j].tolist(), so2)
        errs.append((post - closed).abs().max().item() / post.max().item())
    print(f"  reweighted mixture against grid Bayes per pixel: largest density error {max(errs):.2e} (V {V:.4f}, V' {Vn:.4f})")
    assert max(errs) < 1e-6, errs
    # zero correction: (a) mean_prev = mean; (b) the exact posterior's previous mixture converted to x_t; (c) a constant mixture with K = 1, s = 0
    gm = rand_gm(2, 3, 3, 4, 4, True)
    out, q = G.gm_second_order(gm, G.gm_mean(gm), torch.full((2,), 0.5, dtype=dt), 0.1, 0.1, parts=True)
    assert (q["bias"] == 0).all() and mixture_err(out, gm) < 1e-12, mixture_err(out, gm)
    s0 = s0_map(8).double()
    xp = torch.randn(2, 3, 8, 8, dtype=dt); tp = torch.full((2,), 0.3, dtype=dt)
    gp = {k: v.double() for k, v in exact_posterior_px(xp, tp, s0).items()}
    x = 0.8 * xp + 0.1 * torch.randn_like(xp); tc = torch.full((2,), 0.45, dtype=dt)
    gc = {k: v.double() for k, v in exact_posterior_px(x, tc, s0).items()}
    mp = G.gm_convert_mean(gp, (1 - tp).view(2, 1, 1, 1), (1 - tc).view(2, 1, 1, 1), xp, x)
    _, q = G.gm_second_order(gc, mp, 1 - tc, 0.15, 0.15, parts=True)
    zb = (q["bias"].square().mean() / q["avg_var"].mean()).sqrt().item()
    print(f"  exact posterior: |previous converted - current| / its std {((G.gm_mean(gc) - mp).abs().max() / q['avg_var'].sqrt().min()).item():.1e}; bias RMS / std {zb:.1e}")
    assert zb < 1e-6, zb
    pt = torch.randn(2, 3, 4, 4, dtype=dt)
    const = dict(means=pt[:, None], logweights=torch.zeros(2, 1, 1, 4, 4, dtype=dt), logstd=torch.full((2, 1, 1, 1, 1), -float("inf"), dtype=dt))
    mp = G.gm_convert_mean(const, torch.full((2, 1, 1, 1), 0.7, dtype=dt), torch.full((2, 1, 1, 1), 0.5, dtype=dt), torch.randn_like(pt), torch.randn_like(pt))
    out = G.gm_second_order(const, mp, torch.full((2,), 0.5, dtype=dt), 0.1, 0.2)
    assert torch.allclose(mp, pt, atol=1e-12) and torch.equal(out["means"], const["means"]) and torch.equal(out["logstd"], const["logstd"]), "a constant point mixture must be unchanged"
    # the first call is first order; order 2 then differs; order 1 = c16d71d bitwise; a point model is unchanged by order 2
    oldg = load_at(SPREAD_BASE, "src/diffusion/flow_matching/gmflow.py", "gmflow_c16d71d")
    olds = load_at(SPREAD_BASE, "src/diffusion/flow_matching/sampling_gmflow.py", "sampling_gmflow_c16d71d",
                   drop=("from src.diffusion.flow_matching.gmflow import bridge_coefs, gm_convert_mean, gm_draw, gm_mean\n",),
                   inject={k: getattr(oldg, k) for k in ("bridge_coefs", "gm_convert_mean", "gm_draw", "gm_mean")})
    torch.manual_seed(0)
    nets = {}
    for spread in ("image", "pixel"):
        net = JiT_T2I(**small, gm_k=3, gm_spread=spread)
        for p_ in net.parameters():
            p_.data.add_(torch.randn_like(p_) * 0.05)
        nets[spread] = net
    y = torch.randn(2, 8, 32); mask = torch.ones(2, 8, dtype=torch.bool); mask[:, 4:] = False
    noise = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        for spread, net in nets.items():
            f = lambda x, t, c, **kw: net(x, t, c, y_mask=mask, **kw)
            for cls, kw in ((GMSDESampler, {}), (GMSDESampler, {"final_mean": True}), (GMODESampler, {"substeps": 4})):
                runs = {}
                for tag, c_, extra in (("old", getattr(olds, cls.__name__), {}), ("new1", cls, {}), ("new1x", cls, {"order": 1}), ("new2", cls, {"order": 2})):
                    if tag == "old" and spread == "pixel":
                        continue
                    torch.manual_seed(5)
                    runs[tag] = sampler(c_, 4, rank_seed=False, **kw, **extra)._impl_sampling(f, noise, y, None)
                if "old" in runs:
                    assert all(torch.equal(a_, b_) for a_, b_ in zip(runs["old"][0] + runs["old"][1], runs["new1"][0] + runs["new1"][1])), (cls.__name__, kw, "order 1 differs from " + SPREAD_BASE)
                assert all(torch.equal(a_, b_) for a_, b_ in zip(runs["new1"][0], runs["new1x"][0]))
                assert torch.equal(runs["new1"][0][1], runs["new2"][0][1]), (spread, cls.__name__, "the first call is not first order")
                assert not torch.equal(runs["new1"][0][-1], runs["new2"][0][-1]), (spread, cls.__name__, "order 2 changed nothing")
                assert torch.isfinite(runs["new2"][0][-1]).all()
        for cls in (GMSDESampler, GMODESampler):
            pt_net = lambda x, t, c, gm_mode=None: exact_net(x, t, c)
            torch.manual_seed(5); a1 = sampler(cls, 4, rank_seed=False)._impl_sampling(pt_net, noise, None, None)[0][-1]
            torch.manual_seed(5); a2 = sampler(cls, 4, rank_seed=False, order=2)._impl_sampling(pt_net, noise, None, None)[0][-1]
            assert torch.allclose(a1, a2, atol=1e-6), (cls.__name__, (a1 - a2).abs().max())
    print(f"  order 1 = {SPREAD_BASE} bitwise (GM-SDE, GM-SDE final_mean, GM-ODE); first call of order 2 = first order; a point model is unchanged by order 2")
    # landing with the exact posterior: two-point data (s0 = 0) and per-pixel Gaussian-mixture data
    noise = torch.randn(64, 3, 32, 32)
    for data, s0m in (("two-point", s0_map(32, (0.0,))), ("per-pixel GM", s0_map())):
        for name, cls in (("GM-SDE", GMSDESampler), ("GM-ODE", GMODESampler)):
            for n in (4, 8):
                kw = {"substeps": max(-(-128 // n), 2)} if cls is GMODESampler else {}
                e = {}
                for order in (1, 2):
                    torch.manual_seed(1)
                    xs_, _ = sampler(cls, n, rank_seed=False, order=order, **kw)._impl_sampling(px_net(s0m), noise, None, None)
                    e[order] = landing(xs_[-1], s0m)
                print(f"  {data}, {name} {n} calls: order 1 " + ", ".join(f"{k} {v:.4f}" for k, v in e[1].items()) + " | order 2 " + ", ".join(f"{k} {v:.4f}" for k, v in e[2].items()))
                tol = 0.05 if cls is GMSDESampler else 0.2   # GM-ODE is deterministic: its discretization error shows in a small spread (8 x 16: +14% at s0 = 0.02)
                assert e[2]["near0"] == 1.0 and e[2]["f0"] < 0.015 and all(v < tol for k, v in e[2].items() if k.startswith("std")), (data, name, n, e[2])
                for k_ in e[1]:
                    if k_ != "near0":
                        assert e[2][k_] <= e[1][k_] + {"f0": 0.002, "dist0": 1e-4}.get(k_, 0.005), (data, name, n, k_, e[1][k_], e[2][k_])
    # information only: a network that is not the exact posterior (one Gaussian per pixel with the posterior's mean and variance)
    s0m = s0_map()
    def mm_net(x, t, c, gm_mode="mean"):
        gm = exact_posterior_px(x, t, s0m)
        w = gm["logweights"].exp(); mu = (w * gm["means"]).sum(1, keepdim=True)
        v = (w * (gm["means"] - mu).square()).sum(1).mean(1, keepdim=True).unsqueeze(1) + (2 * G.logstd_x(gm, (1 - t).view(-1, 1, 1, 1, 1))).exp()
        g1 = dict(means=mu, logweights=torch.zeros_like(gm["logweights"][:, :1]), logstd=0.5 * torch.log(v.clamp(min=1e-30)) - torch.log((1 - t).view(-1, 1, 1, 1, 1)))
        return g1 if gm_mode == "full" else mu.squeeze(1)
    for name, cls in (("GM-SDE", GMSDESampler), ("GM-ODE", GMODESampler)):
        for n in (4, 8):
            kw = {"substeps": max(-(-128 // n), 2)} if cls is GMODESampler else {}
            line = []
            for order in (1, 2):
                torch.manual_seed(1)
                xs_, _ = sampler(cls, n, rank_seed=False, order=order, **kw)._impl_sampling(mm_net, noise, None, None)
                e = landing(xs_[-1], s0m)
                line.append(f"order {order}: " + ", ".join(f"{k} {v:.3f}" for k, v in e.items()))
            print(f"  (information) one-Gaussian network, {name} {n} calls: " + " | ".join(line))


if __name__ == "__main__":
    names = sys.argv[1:] or list(RESULTS)
    failed = []
    for name in names:
        print(f"{name}: {RESULTS[name].__doc__.split(chr(10))[0]}", flush=True)
        try:
            RESULTS[name]()
            print(f"{name} PASS", flush=True)
        except Exception:
            traceback.print_exc(limit=3)
            print(f"{name} FAIL", flush=True)
            failed.append(name)
    print("FAILED: " + " ".join(failed) if failed else "ALL PASS")
    sys.exit(len(failed))
