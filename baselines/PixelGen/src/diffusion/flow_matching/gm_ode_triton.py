"""Fused GM-ODE sub-steps (docs/gmflow_hn.md, N): all n analytic sub-steps of one network call in one Triton launch.

The algorithm is GMODESampler's sub-step loop (sampling_gmflow.py) with gm_convert (gmflow.py) written out per pixel. On the sigma grid s_0 > s_1 > ... > s_n of one call
(s_0 = the call's sigma, s_n = the next call's sigma; s_j = s_0 + (s_n - s_0) j / n):
  sub-step 0: x0 = sum_k A_k mu_k;
  sub-step j > 0: the mixture re-weighted to the current x (Eq. 10): with a_s = (1 - s) s_0, b_s = (1 - s_0) s, denom = max(a_s^2 - b_s^2, 1e-6),
    g = (a_s s_0 x - b_s s x_src) / denom, g_var = (s s_0)^2 / denom, v = s_x^2, norm = max(g_var + v, 1e-6),
    mu'_k = (g_var mu_k + v g) / norm, l'_k = log A_k - ||mu_k - g||^2 / (2 norm), x0 = sum_k softmax(l')_k mu'_k;
  each sub-step: x <- x + (s_{j+1} - s_j) (x - x0) / s_j.
Every value is fp32. One program holds BLOCK pixels of one image with all K components in registers, so the means and weights are read once per call and x once.

gm_ode_substeps_ref is the same algorithm in PyTorch (the CPU tests compare it with the sampler's loop and with the kernel under TRITON_INTERPRET=1).
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:   # the CPU tests of the reference need no Triton
    triton = None

CLAMP = 1e-6   # gm_convert's eps for denom and norm


def _check(means, logweights, s_x, x, sigmas):
    B, K, C, H, W = means.shape
    assert C == 3 and x.shape == (B, C, H, W) and logweights.shape == (B, K, 1, H, W), (means.shape, logweights.shape, x.shape)
    assert s_x.shape in ((B, 1, 1, 1, 1), (B, 1, 1, H, W)), s_x.shape
    assert sigmas.ndim == 1 and len(sigmas) >= 2, sigmas.shape


def gm_ode_substeps_ref(means, logweights, s_x, x, sigmas):
    """The reference: the n = len(sigmas) - 1 sub-steps of one call, in fp32 PyTorch. means (B, K, 3, H, W); logweights (B, K, 1, H, W), log-softmax over K;
    s_x (B, 1, 1, 1, 1) or (B, 1, 1, H, W), the image-space std at the call's sigma; x (B, 3, H, W), the state at the call; sigmas (n + 1,) fp32. Returns x after them."""
    _check(means, logweights, s_x, x, sigmas)
    mu, lw, v = means.float(), logweights.float(), s_x.float().square()
    x_src = x.float()
    x = x_src
    s0 = sigmas[0]
    for j in range(len(sigmas) - 1):
        s, s_next = sigmas[j], sigmas[j + 1]
        if j == 0:
            x0 = (lw.exp() * mu).sum(1)
        else:
            a, b = (1 - s) * s0, (1 - s0) * s
            denom = (a * a - b * b).clamp(min=CLAMP)
            g = ((a * s0 * x - b * s * x_src) / denom).unsqueeze(1)
            g_var = (s * s0) * (s * s0) / denom
            norm = (g_var + v).clamp(min=CLAMP)
            logit = lw - 0.5 * (mu - g).square().sum(2, keepdim=True) / norm
            x0 = (logit.softmax(1) * ((g_var * mu + v * g) / norm)).sum(1)
        x = x + (s_next - s) * (x - x0) / s
    return x


if triton is not None:
    @triton.jit
    def _gm_ode_kernel(mu_ptr, lw_ptr, sx_ptr, x_ptr, sig_ptr, out_ptr, HW, sx_sb, sx_sp,
                       N_SUB: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, BLOCK: tl.constexpr):
        b = tl.program_id(1)
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        pm = p < HW
        k = tl.arange(0, KP)
        km = k < K
        m2 = km[:, None] & pm[None, :]
        rk = (b * K + k)[:, None]                                                      # (KP, 1)
        mu0 = tl.load(mu_ptr + (rk * 3 + 0) * HW + p[None, :], mask=m2, other=0.0).to(tl.float32)
        mu1 = tl.load(mu_ptr + (rk * 3 + 1) * HW + p[None, :], mask=m2, other=0.0).to(tl.float32)
        mu2 = tl.load(mu_ptr + (rk * 3 + 2) * HW + p[None, :], mask=m2, other=0.0).to(tl.float32)
        lw = tl.load(lw_ptr + rk * HW + p[None, :], mask=m2, other=0.0).to(tl.float32)
        lw = tl.where(km[:, None], lw, float("-inf"))                                  # padded components weigh 0
        sx = tl.load(sx_ptr + b * sx_sb + p * sx_sp, mask=pm, other=0.0).to(tl.float32)
        v = sx * sx
        xo = b * 3 * HW + p
        xs0 = tl.load(x_ptr + xo, mask=pm, other=0.0).to(tl.float32)
        xs1 = tl.load(x_ptr + xo + HW, mask=pm, other=0.0).to(tl.float32)
        xs2 = tl.load(x_ptr + xo + 2 * HW, mask=pm, other=0.0).to(tl.float32)
        s0 = tl.load(sig_ptr).to(tl.float32)
        # sub-step 0: the mixture mean
        w = tl.exp(lw)
        s_next = tl.load(sig_ptr + 1).to(tl.float32)
        x0 = xs0 + (s_next - s0) * (xs0 - tl.sum(w * mu0, axis=0)) / s0
        x1 = xs1 + (s_next - s0) * (xs1 - tl.sum(w * mu1, axis=0)) / s0
        x2 = xs2 + (s_next - s0) * (xs2 - tl.sum(w * mu2, axis=0)) / s0
        for j in range(1, N_SUB):
            s = tl.load(sig_ptr + j).to(tl.float32)
            s_next = tl.load(sig_ptr + j + 1).to(tl.float32)
            a = (1 - s) * s0
            bb = (1 - s0) * s
            denom = tl.maximum(a * a - bb * bb, 1e-6)
            g_var = (s * s0) * (s * s0) / denom
            g0 = (a * s0 * x0 - bb * s * xs0) / denom
            g1 = (a * s0 * x1 - bb * s * xs1) / denom
            g2 = (a * s0 * x2 - bb * s * xs2) / denom
            norm = tl.maximum(g_var + v, 1e-6)                                         # (BLOCK,)
            d0 = mu0 - g0[None, :]
            d1 = mu1 - g1[None, :]
            d2 = mu2 - g2[None, :]
            logit = lw - 0.5 * (d0 * d0 + d1 * d1 + d2 * d2) / norm[None, :]
            e = tl.exp(logit - tl.max(logit, axis=0)[None, :])
            z = tl.sum(e, axis=0)
            e0 = tl.sum(e * ((g_var * mu0 + v[None, :] * g0[None, :]) / norm[None, :]), axis=0) / z
            e1 = tl.sum(e * ((g_var * mu1 + v[None, :] * g1[None, :]) / norm[None, :]), axis=0) / z
            e2 = tl.sum(e * ((g_var * mu2 + v[None, :] * g2[None, :]) / norm[None, :]), axis=0) / z
            x0 = x0 + (s_next - s) * (x0 - e0) / s
            x1 = x1 + (s_next - s) * (x1 - e1) / s
            x2 = x2 + (s_next - s) * (x2 - e2) / s
        tl.store(out_ptr + xo, x0, mask=pm)
        tl.store(out_ptr + xo + HW, x1, mask=pm)
        tl.store(out_ptr + xo + 2 * HW, x2, mask=pm)


def gm_ode_substeps_fused(means, logweights, s_x, x, sigmas, block=256, num_warps=4):
    """The kernel: one launch for all sub-steps of one call (the sub-step count is a compile-time constant: one compile per count). Same arguments and result as gm_ode_substeps_ref. Runs on CUDA tensors, or on CPU tensors when
    TRITON_INTERPRET=1 was set before Triton was imported."""
    assert triton is not None, "Triton is not installed"
    _check(means, logweights, s_x, x, sigmas)
    B, K, C, H, W = means.shape
    HW = H * W
    means, logweights, x = means.contiguous(), logweights.contiguous(), x.float().contiguous()
    sx = s_x.float().contiguous()
    sx_sb, sx_sp = (1, 0) if sx.shape[-1] * sx.shape[-2] == 1 else (HW, 1)
    sig = sigmas.float().contiguous().to(x.device)
    out = torch.empty_like(x)
    grid = (triton.cdiv(HW, block), B)
    _gm_ode_kernel[grid](means, logweights, sx, x, sig, out, HW, sx_sb, sx_sp,
                         N_SUB=len(sig) - 1, K=K, KP=max(2, triton.next_power_of_2(K)), BLOCK=block, num_warps=num_warps)
    return out
