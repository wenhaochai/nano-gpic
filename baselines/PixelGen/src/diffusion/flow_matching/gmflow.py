"""GMFlow math in our convention (docs/gmflow.md, docs/gmflow_digest_2026-11-05.md).

Our t = 1 is data and x_t = t x + (1 - t) eps. With sigma := 1 - t and alpha := t, GMFlow's formulas carry over unchanged.
A mixture is a dict: means (B, K, C, H, W), the x-means mu_x,k; logweights (B, K, 1, H, W), log-softmax over K; logstd (B, 1, 1, 1, 1), log s_u in
velocity units, so the image-space std is s_x = sigma s_u. With the per-pixel spread (JiT_T2I gm_spread "pixel") logstd is (B, 1, 1, H, W): one s_u per pixel, shared by
the C channels and the K components. Every function here takes either shape; every reduction over pixels is noted where it happens. Every function runs in fp32 with
autocast off.
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
    """log s_x = log s_u + log sigma, (B, 1, 1, 1, 1) or (B, 1, 1, H, W). sigma (B, 1, 1, 1, 1)."""
    return gm["logstd"] + torch.log(sigma.clamp(min=EPS))


def image_scope(gm):
    """True for one mixture per image (JiT_T2I gm_scope "image", docs/gmflow.md section 15): logweights (B, K, 1, 1, 1) over means of a larger image."""
    lw, m = gm["logweights"], gm["means"]
    return lw.shape[-2:] == (1, 1) and m.shape[-2:] != (1, 1)


def gm_component_ll(target, means, logstd):
    """log N(target; means_k, exp(logstd)^2 I) of each component at each pixel, summed over the C channels, without the constant C/2 log 2 pi. Returns (B, K, H, W)."""
    inverse_std = torch.exp(-logstd).clamp(max=1 / EPS)
    diff = (means - target.unsqueeze(1)) * inverse_std
    return (-0.5 * diff.square() - logstd).sum(dim=2)


def gm_nll(target, means, logweights, logstd, tau=1.0):
    """Per-pixel -log sum_k A_k N(target; means_k, exp(logstd)^2 I) over the C channels, without the constant C/2 log 2 pi (GMFlow diffusion_loss.py:33-36).
    target (B, C, H, W); means (B, K, C, H, W); logweights (B, K, 1, H, W); logstd broadcastable to (B, K, 1, H, W). Returns (B, H, W).
    One mixture per image (logweights (B, K, 1, 1, 1), docs/gmflow.md section 15): each component scores the whole image, -log sum_k A_k prod_pixels N(...),
    divided by the number of pixels, returned as (B, 1, 1), so a mean over the pixels gives the same scale as the per-pixel loss. With K = 1 the two agree.
    tau (one mixture per image only; docs/gmflow.md section 16): deterministic annealing EM (Ueda and Nakano 1998) of the posterior over the K candidates,
    -(tau / pixels) log sum_k (A_k prod_pixels N(...))^(1 / tau). The gradient on the means is that of tau = 1 with the responsibilities
    softmax_k((log A_k + ll_k) / tau) in place of the exact ones, so a large tau shares it among the candidates, and the gradient on the logits is
    (A_k - r_k(tau)) / pixels at every tau. With A_k outside the temperature (Rose's mass-constrained form) that gradient would grow with the gaps in ll_k and
    Adam would move the weights to the best candidate, which the means' gradient (r ~ A at a large tau) then follows. tau = 1 is the loss above, the same expression."""
    ll = gm_component_ll(target, means, logstd)                                         # (B, K, H, W)
    lw = logweights.squeeze(2)                                                         # (B, K, H, W) or (B, K, 1, 1)
    if lw.shape[-2:] == (1, 1) and ll.shape[-2:] != (1, 1):
        n_pix = ll.shape[-2] * ll.shape[-1]
        if tau == 1:
            return (-torch.logsumexp(ll.sum((-2, -1)) + lw.flatten(1), dim=1) / n_pix).view(-1, 1, 1)
        return (-tau * torch.logsumexp((ll.sum((-2, -1)) + lw.flatten(1)) / tau, dim=1) / n_pix).view(-1, 1, 1)
    assert tau == 1, "the temperature tempers the posterior over whole candidate images: one mixture per image only"
    return -torch.logsumexp(ll + lw, dim=1)


def image_resp(target, means, logweights, logstd, tau=1.0):
    """One mixture per image: the responsibilities of the K candidates that weight the gradient of the means under the temperature tau,
    softmax_k((log A_k + ll_k) / tau) (section 16; tau = 1: the exact posterior). Returns (B, K)."""
    ll = gm_component_ll(target, means, logstd).sum((-2, -1))                          # (B, K)
    return ((ll + logweights.flatten(1)) / tau).softmax(1)


def image_wins(target, means, logweights, logstd):
    """One mixture per image: for each image the component with the highest posterior responsibility (the one that gets the gradient of the means). Returns (B,)."""
    ll = gm_component_ll(target, means, logstd).sum((-2, -1))                          # (B, K)
    return (ll + logweights.flatten(1)).argmax(1)


def bridge_mixture(gm, x_high, sigma_high, sigma_low):
    """The mixture over x_low that the transition loss scores: means c1 x_high + c2 mu_x,k and log std 0.5 log(c2^2 s_x^2 + c3) (GMFlow gmflow.py:192-204, x0 branch).
    sigmas (B, 1, 1, 1). Returns (means (B, K, C, H, W), logstd)."""
    c1, c2, c3 = bridge_coefs(sigma_high, sigma_low)
    c1, c2, c3 = c1.unsqueeze(1), c2.unsqueeze(1), c3.unsqueeze(1)                     # (B, 1, 1, 1, 1)
    lsx = logstd_x(gm, sigma_high.unsqueeze(1))
    # log(0) = -inf when c3 = 0 (lambda = 1), and logaddexp then returns the first term
    logstd_low = torch.logaddexp(2 * (lsx + torch.log(c2.clamp(min=1e-6))), torch.log(c3.clamp(min=0))) / 2
    return c1 * x_high.unsqueeze(1) + c2 * gm["means"], logstd_low


def transition_nll(gm, x_low, x_high, sigma_high, sigma_low, tau=1.0):
    """-log q(x_low | x_high) per pixel, q = sum_k A_k N(c1 x_high + c2 mu_x,k, (c2^2 s_x^2 + c3) I) (GMFlow gmflow.py:192-204, x0 branch). sigmas (B, 1, 1, 1).
    tau: the temperature of one mixture per image (gm_nll)."""
    means, logstd_low = bridge_mixture(gm, x_high, sigma_high, sigma_low)
    return gm_nll(x_low, means, gm["logweights"], logstd_low, tau)


def transition_wins(gm, x_low, x_high, sigma_high, sigma_low):
    """One mixture per image: the winning component of each image under the transition loss's mixture (as transition_nll). Returns (B,)."""
    means, logstd_low = bridge_mixture(gm, x_high, sigma_high, sigma_low)
    return image_wins(x_low, means, gm["logweights"], logstd_low)


def transition_resp(gm, x_low, x_high, sigma_high, sigma_low, tau=1.0):
    """One mixture per image: the responsibilities under the transition loss's mixture and the temperature tau (image_resp). Returns (B, K)."""
    means, logstd_low = bridge_mixture(gm, x_high, sigma_high, sigma_low)
    return image_resp(x_low, means, gm["logweights"], logstd_low, tau)


def gm_mean(gm):
    """sum_k A_k mu_x,k, (B, C, H, W)."""
    return (gm["logweights"].exp() * gm["means"]).sum(1)


def gm_draw(gm, sigma):
    """One x0 per pixel, independently: a component by its weights (Gumbel-max), then + s_x xi on the C channels (GMFlow gmflow_ops.py:637-655). sigma (B, 1, 1, 1).
    Not for one mixture per image: its (B, K, 1, 1, 1) weights would gather one pixel only (gm_pick draws whole images)."""
    assert not image_scope(gm), "gm_draw draws per pixel; use gm_pick for one mixture per image"
    lw = gm["logweights"].squeeze(2)                                                    # (B, K, H, W)
    u = torch.rand_like(lw).clamp(min=1e-12, max=1 - 1e-7)   # fp32: 1 - 1e-7 < 1, so -log(u) > 0
    gumbel = -torch.log(-torch.log(u))
    k = (lw + gumbel).argmax(1, keepdim=True)                                           # (B, 1, H, W)
    means = gm["means"]
    mu = means.gather(1, k.unsqueeze(2).expand(-1, -1, means.shape[2], -1, -1)).squeeze(1)
    return mu + logstd_x(gm, sigma.unsqueeze(1)).squeeze(1).exp() * torch.randn_like(mu)


def gm_pick(gm):
    """One mixture per image (docs/gmflow.md section 15, the owner's sampler): one component per image by its weights (Gumbel-max), and that whole candidate image,
    with no noise added. Returns (B, C, H, W)."""
    assert image_scope(gm), "gm_pick is for one mixture per image"
    lw = gm["logweights"].flatten(1)                                                    # (B, K)
    u = torch.rand_like(lw).clamp(min=1e-12, max=1 - 1e-7)
    k = (lw - torch.log(-torch.log(u))).argmax(1)                                       # (B,)
    return gm["means"][torch.arange(k.shape[0], device=k.device), k]


def gm_convert(gm, sigma_src, sigma_tgt, x_src, x_tgt, eps=1e-6):
    """The analytic sub-step (GMFlow Eq. 10; code gmflow.py:41-70): q(x0 | x_tgt) proportional to p(x_tgt | x0) / p(x_src | x0) q(x0 | x_src), for sigma_tgt < sigma_src.
    q(x0 | x_src) is gm with s_x = sigma_src s_u. Returns (means, logweights, var) of the new mixture: (B, K, C, H, W), (B, K, 1, H, W), and the shape of gm's logstd.
    Pixel by pixel: nothing here reduces over pixels, so a per-pixel s_x gives each pixel its own Bayes update (tools/gmflow/test_gmflow.py T13).
    Not for one mixture per image: its update would need the whole image's likelihood, so it refuses."""
    assert not image_scope(gm), "gm_convert updates per pixel; one mixture per image has no sub-step here (docs/gmflow.md section 15)"
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


GM2_CA = 0.005   # c_a of GM-SDE 2 / GM-ODE 2 (GMFlow forward_test gm2_coefs[0]; paper App. A.2)


def gm_iso_var(gm, sigma):
    """The isotropic surrogate's variance of the x-mixture (GMFlow gm_to_iso_gaussian, then .mean over H, W as gm_2nd_order's no-guidance branch does): per pixel,
    sum_k A_k |mu_k - mean|^2 / C + s_x^2, then the average over the pixels, (B, 1, 1, 1). A per-pixel s_x enters through that average: the surrogate is isotropic over
    the whole image, s^2 I in D = C H W dimensions, as the paper's eq:gm2. sigma (B, 1, 1, 1)."""
    w = gm["logweights"].exp()
    mean = (w * gm["means"]).sum(1, keepdim=True)                                        # (B, 1, C, H, W)
    var = (w * (gm["means"] - mean).square()).sum(1).mean(1, keepdim=True)               # (B, 1, H, W): over K, then over C
    var = var + (2 * logstd_x(gm, sigma.unsqueeze(1))).exp().squeeze(1)                 # + s_x^2, (B, 1, 1, 1) or (B, 1, H, W)
    return mean.squeeze(1), var.mean((-1, -2), keepdim=True)


def gm_second_order(gm, mean_prev, sigma, h, prev_h, ca=GM2_CA, parts=False):
    """GM-SDE 2 / GM-ODE 2 without guidance (paper App. A.2, eq:gm2; GMFlow gmflow.py:251-291, the branch cfg_bias is None). gm: this call's mixture at sigma (B,);
    mean_prev: the previous call's raw mixture converted to this x_t (Eq. 10, gm_convert_mean); h, prev_h: this and the previous call's step in sigma (floats; the
    code's h / prev_h and prev_h / num_timesteps). As the code:
      mean, var = the isotropic surrogate of gm (gm_iso_var: averaged over the pixels); avg_var = var;
      k = 0.5 h / prev_h; err_power = avg_var c_a; mean_diff = (mean - mean_prev) sqrt(max(0, 1 - err_power / prev_h^2)) k;
      bias = mean_diff with its power (mean square over C, H, W) capped at avg_var; v' = var max(1e-6, 1 - bias_power / avg_var), bias_power before the cap;
      gm <- gm N(x; mean + bias, v') / N(x; mean, var).
    The code forms the Gaussian ratio (precision beta / v', beta = 1 - v' / var) as N(r_mean, r_var) with r_var = var v' / (var - v') and multiplies each component by it,
    with clamp(min=1e-6) on var - v' and on r_var + s_x^2. Here the same product is written with no division by var - v':
      mu_k' = (mu_k v' + s_x^2 (bias + beta mean)) / (v' + s_x^2 beta),   s_x'^2 = s_x^2 v' / (v' + s_x^2 beta),
      log A_k' = log A_k - (beta |mu_k - mean|^2 - 2 bias . (mu_k - mean)) / (2 (v' + s_x^2 beta)) + const over k.
    It equals the code wherever its two clamps are inactive (test T14); where var - v' < 1e-6 (bias_power < 1e-6 avg_var) the code's clamp pulls every mean towards 0
    with variance var^2 / 1e-6, and this form keeps the exact ratio (a no-op at bias = 0). An image with var <= 1e-12 (every pixel nearly certain) is returned unchanged:
    its bias is capped under 1e-6 RMS. (The code there divides by its clamps: with s_x = 0 and var ~ 0 its means collapse to 0.) A per-pixel s_x enters through var (an image average) and, pixel by pixel, through s_x^2 in the product.
    Returns the new mixture (logstd in velocity units, the shape of gm's) and, with parts, the quantities."""
    sig_b = sigma.view(-1, 1, 1, 1)
    mean, var = gm_iso_var(gm, sig_b)                                                    # (B, C, H, W), (B, 1, 1, 1)
    avg_var = var.mean((-3, -2, -1), keepdim=True)                                       # the code keeps both; equal here
    k = 0.5 * h / prev_h
    err_power = avg_var * ca
    scale = (1 - err_power / (prev_h * prev_h)).clamp(min=0).sqrt() * k
    mean_diff = (mean - mean_prev) * scale
    bias_power = mean_diff.square().mean((-3, -2, -1), keepdim=True)
    bias = mean_diff * (avg_var / bias_power.clamp(min=1e-6)).clamp(max=1).sqrt()
    rho = (1 - bias_power / avg_var.clamp(min=1e-6)).clamp(min=1e-6)
    new_var, beta = var * rho, 1 - rho
    lsx = logstd_x(gm, sig_b.unsqueeze(1))
    g = (2 * lsx).exp()                                                                   # s_x^2, (B, 1, 1, 1, 1) or (B, 1, 1, H, W)
    v5, b5, beta5, m5 = new_var.unsqueeze(1), bias.unsqueeze(1), beta.unsqueeze(1), mean.unsqueeze(1)
    denom = v5 + g * beta5
    d = gm["means"] - m5
    means = (gm["means"] * v5 + g * (b5 + beta5 * m5)) / denom
    logweights = (gm["logweights"] - (beta5 * d.square() - 2 * b5 * d).sum(2, keepdim=True) / (2 * denom)).log_softmax(1)
    lsx_new = lsx + 0.5 * (torch.log(v5) - torch.log(denom))
    ok = (var > 1e-12).unsqueeze(1)   # (B, 1, 1, 1, 1); below it the bias is capped under 1e-6 RMS and v' could underflow
    out = dict(means=torch.where(ok, means, gm["means"]), logweights=torch.where(ok, logweights, gm["logweights"]),
               logstd=torch.where(ok, lsx_new, lsx) - torch.log(sig_b.unsqueeze(1).clamp(min=EPS)))
    if parts:
        return out, dict(k=k, err_power=err_power, scale=scale, mean=mean, mean_diff=mean_diff, bias_power=bias_power, bias=bias, avg_var=avg_var, var=var,
                         new_var=new_var)
    return out
