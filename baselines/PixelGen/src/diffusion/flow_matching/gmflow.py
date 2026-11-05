"""GMFlow math in our convention (docs/gmflow.md, docs/gmflow_digest_2026-11-05.md).

Our t = 1 is data and x_t = t x + (1 - t) eps. With sigma := 1 - t and alpha := t, GMFlow's formulas carry over unchanged.
A mixture is a dict: means (B, K, C, H, W), the x-means mu_x,k; logweights (B, K, 1, H, W), log-softmax over K; logstd (B, 1, 1, 1, 1), log s_u in
velocity units, so the image-space std is s_x = sigma s_u. Every function here runs in fp32 with autocast off.
"""
import torch

EPS = 1e-4   # GMFlow's training eps: sigma >= EPS, inverse std <= 1 / EPS (lib/models/losses/diffusion_loss.py:33, gmflow.py:392-395)


def bridge_coefs(sigma, sigma_to, eps=1e-6):
    """p(x_{sigma_to} | x_sigma, x0) = N(c1 x_sigma + c2 x0, c3 I) for sigma_to < sigma (GMFlow gmflow.py:185-191, gmflow_sde.py:126-141)."""
    alpha, alpha_to = 1 - sigma, 1 - sigma_to
    r = sigma_to / sigma.clamp(min=eps)
    a = alpha / alpha_to.clamp(min=eps)
    beta_over_sigma_sq = 1 - (r * a) ** 2
    c1 = r ** 2 * a
    c2 = beta_over_sigma_sq * alpha_to
    c3 = beta_over_sigma_sq * sigma_to ** 2
    return c1, c2, c3


def forward_transition(x_low, sigma_low, sigma_high, noise):
    """x_{sigma_high} from x_{sigma_low} (GMFlow gmflow.py:77-91)."""
    mean_trans = (1 - sigma_high) / (1 - sigma_low)
    std_trans = (sigma_high ** 2 - (mean_trans * sigma_low) ** 2).clamp(min=0).sqrt()
    return x_low * mean_trans + noise * std_trans


def logstd_x(gm, sigma):
    """log s_x = log s_u + log sigma, (B, 1, 1, 1, 1)."""
    return gm["logstd"] + torch.log(sigma.clamp(min=EPS))


def gm_nll(target, means, logweights, logstd):
    """Per-pixel -log sum_k A_k N(target; means_k, exp(logstd)^2 I) over the C channels, without the constant C/2 log 2 pi (GMFlow diffusion_loss.py:33-36).
    target (B, C, H, W); means (B, K, C, H, W); logweights (B, K, 1, H, W); logstd broadcastable to (B, K, 1, H, W). Returns (B, H, W)."""
    inverse_std = torch.exp(-logstd).clamp(max=1 / EPS)
    diff = (means - target.unsqueeze(1)) * inverse_std
    ll = (-0.5 * diff.square() - logstd).sum(dim=2)          # (B, K, H, W)
    return -torch.logsumexp(ll + logweights.squeeze(2), dim=1)


def transition_nll(gm, x_low, x_high, sigma_high, sigma_low):
    """-log q(x_low | x_high) per pixel, q = sum_k A_k N(c1 x_high + c2 mu_x,k, (c2^2 s_x^2 + c3) I) (GMFlow gmflow.py:192-204, x0 branch). sigmas (B, 1, 1, 1)."""
    c1, c2, c3 = bridge_coefs(sigma_high, sigma_low)
    c1, c2, c3 = c1.unsqueeze(1), c2.unsqueeze(1), c3.unsqueeze(1)                     # (B, 1, 1, 1, 1)
    lsx = logstd_x(gm, sigma_high.unsqueeze(1))
    # log std of the transition: 0.5 log(c2^2 s_x^2 + c3); log(0) = -inf when c3 = 0 (lambda = 1), and logaddexp then returns the first term
    logstd_low = torch.logaddexp(2 * (lsx + torch.log(c2.clamp(min=1e-6))), torch.log(c3.clamp(min=0))) / 2
    return gm_nll(x_low, c1 * x_high.unsqueeze(1) + c2 * gm["means"], gm["logweights"], logstd_low)


def gm_mean(gm):
    """sum_k A_k mu_x,k, (B, C, H, W)."""
    return (gm["logweights"].exp() * gm["means"]).sum(1)


def gm_draw(gm, sigma):
    """One x0 per pixel, independently: a component by its weights (Gumbel-max), then + s_x xi on the C channels (GMFlow gmflow_ops.py:637-655). sigma (B, 1, 1, 1)."""
    lw = gm["logweights"].squeeze(2)                                                    # (B, K, H, W)
    u = torch.rand_like(lw).clamp(min=1e-12, max=1 - 1e-7)   # fp32: 1 - 1e-7 < 1, so -log(u) > 0
    gumbel = -torch.log(-torch.log(u))
    k = (lw + gumbel).argmax(1, keepdim=True)                                           # (B, 1, H, W)
    means = gm["means"]
    mu = means.gather(1, k.unsqueeze(2).expand(-1, -1, means.shape[2], -1, -1)).squeeze(1)
    return mu + logstd_x(gm, sigma.unsqueeze(1)).squeeze(1).exp() * torch.randn_like(mu)


def gm_convert(gm, sigma_src, sigma_tgt, x_src, x_tgt, eps=1e-6):
    """The analytic sub-step (GMFlow Eq. 10; code gmflow.py:41-70): q(x0 | x_tgt) proportional to p(x_tgt | x0) / p(x_src | x0) q(x0 | x_src), for sigma_tgt < sigma_src.
    q(x0 | x_src) is gm with s_x = sigma_src s_u. Returns (means, logweights, var) of the new mixture: (B, K, C, H, W), (B, K, 1, H, W), (B, 1, 1, 1, 1)."""
    alpha_src, alpha_tgt = 1 - sigma_src, 1 - sigma_tgt
    ats, ast = alpha_tgt * sigma_src, alpha_src * sigma_tgt
    denom = (ats.square() - ast.square()).clamp(min=eps)
    g_mean = ((ats * sigma_src * x_tgt - ast * sigma_tgt * x_src) / denom).unsqueeze(1)  # (B, 1, C, H, W)
    g_var = ((sigma_tgt * sigma_src).square() / denom).unsqueeze(1)                       # (B, 1, 1, 1, 1)
    gm_var = (2 * logstd_x(gm, sigma_src.unsqueeze(1))).exp()
    norm = (g_var + gm_var).clamp(min=eps)
    means = (g_var * gm["means"] + gm_var * g_mean) / norm
    logweights = (gm["logweights"] + (gm["means"] - g_mean).square().sum(2, keepdim=True) * (-0.5 / norm)).log_softmax(1)
    return means, logweights, g_var * gm_var / norm


def gm_convert_mean(gm, sigma_src, sigma_tgt, x_src, x_tgt):
    means, logweights, _ = gm_convert(gm, sigma_src, sigma_tgt, x_src, x_tgt)
    return (logweights.exp() * means).sum(1)
