#!/usr/bin/env python3
"""CPU tests of the GMFlow head (docs/gmflow.md): JiT_T2I(gm_k > 0), GMFlowTrainer (training_gmflow.py), GMSDESampler / GMODESampler (sampling_gmflow.py).
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
        hand = 2 * 256 * 640 * (K * 4 * 256 - 768) + 2 * (640 * 1024 + 1024 * 1)
        assert gm - point == hand, (K, gm - point, hand)
        mean_mode = fwd_flops(JiT_T2I(**d640, gm_k=K))
        r = (gm - point) / REF
        steps = int(BASE_STEPS / (1 + r)) // 128 * 128
        while (steps + 128) * (1 + r) <= BASE_STEPS:
            steps += 128
        assert steps * (1 + r) <= BASE_STEPS
        print(f"  K={K}: point {point / 1e9:.4f} GFLOP, GMFlow {gm / 1e9:.4f} (mean mode {mean_mode / 1e9:.4f}); added {(gm - point) / 1e6:.2f} MFLOP = r {r:.5f} of 34.72; STEPS {steps} (x (1 + r) = {steps * (1 + r):.0f} <= {BASE_STEPS})")


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
