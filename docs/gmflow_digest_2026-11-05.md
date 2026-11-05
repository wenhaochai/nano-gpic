# GMFlow digest for a port (2026-11-05)

**Sources.**
- Paper: Hansheng Chen, Kai Zhang, Hao Tan, Zexiang Xu, Fujun Luan, Leonidas Guibas, Gordon Wetzstein, Sai Bi, "Gaussian Mixture Flow Matching Models", ICML 2025, **arXiv 2504.05304v3**. The ID was checked against the title through the arXiv API. I read the LaTeX source (`https://arxiv.org/e-print/2504.05304v3`).
- Code: **github.com/Lakonik/GMFlow** (the arXiv comment links it), commit `c3697df3c83db9e3f4d89f1ce3b8f802f9136160` (2026-08-04).
- Follow-up: "pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation", **arXiv 2510.14974v3**, LaTeX source. I read it only for facts about the head, factorization, variance and temperature.
- Local copies (untrusted, read only, never run): `/scratch/gpfs/GROUP/USER/tmp/gmflow_ref/{paper_src,repo,pi_src,issues}`.

**Citation conventions.**
- Equation numbers (1)–(10) are the paper's main-text numbers. I counted them from the source; every displayed equation in Secs. 2–3 has a label.
- Appendix equations are cited by section and LaTeX label, because the appendix numbering depends on the compiled file.
- Figure and table numbers follow the order of the source: Fig. 6 = `fig:comparisonplot`, Fig. 7 = `fig:ksweepplot`, Tab. 2 = `tab:guidance`, Tab. 3 = `tab:likelihood`, Tab. 4 = `tab:ablation`, Tab. 5 = `tab:comp_gms`.
- Code paths are relative to the repo root. Line numbers refer to commit `c3697df`.

**Time convention used everywhere below.** GMFlow uses $t=0$ for data and $t=1$ for noise: $\alpha_t = 1-t$, $\sigma_t = t$, $\vx_t = \alpha_t\vx_0 + \sigma_t\epsilon$. Our JiT convention is the reverse: $t=1$ is data. Sec. 12 gives the mapping.

---

## 1. Parameterization

**Paper (Sec. 3.1, Eq. 4):**
$$q_{\theta}(\mathbf u\mid \mathbf x_t)=\sum_{k=1}^K A_k\,\mathcal N(\mathbf u;\boldsymbol\mu_k,\boldsymbol\Sigma_k),\qquad A_k=\frac{\exp a_k}{\sum_k \exp a_k}$$
"Simplified covariances" (Sec. 3.1): $\boldsymbol\Sigma_k = s^2\mathbf I$, "where $s\in\mathbb R_+$ is the predicted standard deviation shared by all mixture components."

**What the means predict.** The means predict the **random velocity $\mathbf u$** (Eq. 2): $\mathbf u := (\mathbf x_t-\mathbf x_0)/\sigma_t$. Under the linear schedule this equals $\epsilon - \mathbf x_0$.
- Code: `lib/models/diffusions/gaussian_flow.py:91`, `loss_kwargs.update(u_t=noise - x_0)`.
- The $\mathbf x_0$ form is derived from it, never predicted directly (Sec. 3.1, "u-to-x0 reparameterization"): $\boldsymbol\mu_{x,k}=\mathbf x_t-\sigma_t\boldsymbol\mu_k$ and $s_x=\sigma_t s$.
- Code: `lib/models/diffusions/gmflow.py:105-106`:
  ```python
  means_x_0 = x_t - sigma * denoising_output['means']
  logstds_x_0 = denoising_output['logstds'] + torch.log(sigma.clamp(min=eps))
  ```

**Granularity: per latent pixel, over that pixel's C channels.**
- Paper, Sec. 3.4 "Pixel-wise factorization": "we treat each pixel (in the latent grid for latent diffusion) as an independent low-dimensional GM, making the training loss the sum of per-pixel GM KL terms."
- pi-Flow, Sec. 3 ("GMFlow Policy") writes it as $q(\mathbf u\mid\mathbf x)=\prod_{i=1}^{L}\sum_k A_{ik}\mathcal N(\mathbf u_i;\boldsymbol\mu_{ik},s^2\mathbf I)$, with $\boldsymbol\mu_{ik}\in\mathbb R^C$ and $L\times C = D$.
- The DiT emits $p^2\cdot K(C+1)$ values per token. The code then unpatchifies them, so each latent pixel (not each patch) gets its own K means and its own K logits. In `lib/models/architecture/gmflow.py`:
  ```python
  261  self.gm_channels = num_gaussians * (self.out_channels + 1)
  296  self.proj_out_2 = nn.Linear(
  297      self.inner_dim, self.config.patch_size * self.config.patch_size * self.gm_channels)
  398  hidden_states = self.proj_out_2(hidden_states).reshape(
  399          bs, height, width, self.patch_size, self.patch_size, self.gm_channels
  400      ).permute(0, 5, 1, 3, 2, 4).reshape(
  401          bs, self.gm_channels, height * self.patch_size, width * self.patch_size)
  ```

**Means and weights.** In `lib/models/architecture/gmflow.py:203-205`:
```python
means, logweights = x.split([self.num_gaussians * self.out_channels, self.num_gaussians], dim=1)
means = means.view(bs, self.num_gaussians, self.out_channels, h, w)
logweights = logweights.view(bs, self.num_gaussians, 1, h, w).log_softmax(dim=1)
```
- The softmax runs over the **K axis**, separately for each pixel.
- The pixel's C channels share one weight vector, because each component is a C-dimensional Gaussian.
- Training has **no temperature**. pi-Flow adds an inference-only temperature for its distilled policy (pi-Flow App. "GM Temperature"): $q_T \approx \sum_k \frac{A_k^{1/T}}{\sum_z A_z^{1/T}}\mathcal N(\mathbf u;\boldsymbol\mu_k, s^2T\mathbf I)$. It uses T = 0.3 at 4 NFE and T = 0.7 at 8 NFE, and no temperature on the final step. GMFlow itself does not use T.

**Variance.**
- The variance is **isotropic** and **shared across the K components, all pixels and all channels**. It is a **single scalar per sample**.
- It is predicted by a separate small MLP from the **time + class embedding**, with the gradient into that embedding stopped.
- Paper, Sec. 4.2: "a tiny two-layer MLP separately predicts $s$ using only the time $t$ and condition $\mathbf c$ as inputs. Empirically, we find this design to be more stable than predicting $s$ using the main DiT."
- Code, `lib/models/architecture/gmflow.py`:
  ```python
  183  logstd_layers.extend([act(), nn.Linear(in_dim, logstd_inner_dim)])   # 1152 -> 1024, SiLU first
  187  self.logstd_layers = nn.Sequential(*logstd_layers, act(), nn.Linear(in_dim, 1))
  207  logstds = self.logstd_layers(emb).view(bs, 1, 1, 1, 1)
  403  return self.gm_out(hidden_states, cond_emb.detach())
  ```
  `cond_emb` is the DiT `CombinedTimestepLabelEmbeddings` output (timestep + class label), `gmflow.py:341-342`.
- A fixed-variance option exists: `constant_logstd` (`gmflow.py:161,169-171,208-211`). The released configs do not use it.

**Floor or clamp on log s.**
- The head has none. The raw linear output is used as $\log s$.
- The loss clamps the inverse std at $1/\epsilon = 10^4$ (`lib/models/losses/diffusion_loss.py:33`, `inverse_std = torch.exp(-pred_logstds).clamp(max=1 / eps)`, with eps = 1e-4).
- Below $\log s \approx -9.21$ the quadratic term stops shrinking. The $+\log s$ term keeps pulling, however, so this clamp is not a true floor.
- The transition loss (Sec. 3) adds a real variance floor $c_3$ on the training target.

**Initialization** (`lib/models/architecture/gmflow.py:194-199, 308-328`):
```python
198  xavier_init(m, distribution='uniform')        # every nn.Linear, including proj_out_2 (means + logits)
199  constant_init(self.logstd_layers[-1], val=0)  # log s = 0 at init, so s = 1
323  constant_init(m.linear, val=0)                # adaLN-Zero in blocks
326  constant_init(self.proj_out_1, val=0)         # final adaLN shift/scale
```
- In the released model (V1, `GMDiTTransformer2DModel`, used by the configs and by `lib/pipelines/gmdit_pipeline.py:11`), **`proj_out_2` is not zero-initialized**. The means and the logits therefore start as random Xavier projections of the normalized features: the logits are not uniform and the means are spread out.
- An unused variant `GMDiTTransformer2DModelV2` splits the heads and zero-inits the logits and the log s output (`gmflow.py:569-570`). The means stay Xavier.
- The 2-D toy model zero-inits the logit and log s weights and uses Kaiming init for the means (`lib/models/architecture/toymodels.py:90-97`).

**Paper vs. config on initialization.** Paper Sec. 4.2: "We train a DiT-XL/2 ... as the baseline, and then adapt it into GMFlow-DiT by expanding its output channels and training it with the transition loss." The released training config has `load_from = None` and no `pretrained` key (`configs/gmflow_imagenet_k8_8gpus.py:155`), so it trains from scratch. Neither source says whether the paper's GMFlow model was warm-started from the baseline weights.

**Noise schedule.** Linear (rectified flow), $\alpha_t=1-t$, $\sigma_t=t$, T = 1 (Sec. 2, "flow matching models ... define $T:=1, \alpha_t:=1-t, \sigma_t:=t$").
- Code works in $t\in[0,1000]$ with `sigma = t / num_timesteps`: `gaussian_flow.py:59-61`, `std = t.reshape(...) / self.num_timesteps; mean = 1 - std; return x_0 * mean + noise * std`.

## 2. Main ImageNet model: sizes and overhead

| Item | Value | Source |
|---|---|---|
| Backbone | DiT-XL/2: 28 layers, 16 heads × 72 = width 1152 | `configs/gmflow_imagenet_k8_8gpus.py:18-21`; Sec. 4.2 |
| Patch | 2 | `gmflow.py:237` default `patch_size=2` |
| Latent | 4 × 32 × 32 (SD-VAE `stabilityai/sd-vae-ft-ema`, 256²) | config `:7-8, 22, 57` |
| K | 8 | config `:15` |
| Values per latent pixel | K·(C+1) = 8·5 = 40 (32 means + 8 logits) | `gmflow.py:261` |
| Final linear | 1152 → 2·2·40 = 160 per token | `gmflow.py:296-297` |
| s-MLP | SiLU → Linear(1152, 1024) → SiLU → Linear(1024, 1) | config `:16-17`; `gmflow.py:179-190` |
| Spectrum MLP (SDE only) | Linear(2, 64) → SiLU → Linear(64, 8) → SiLU → Linear(8, 4·32·32) | config `:25-29`; `gmflow.py:702-757` |

**Overhead.**
- Parameters: "The adaptation results in only a 0.2% increase in network parameters for K = 8" (Sec. 4.2).
- Time: "it adds only 0.005 sec of overhead per step (batch size 125, A100 GPU) compared to its flow-matching counterpart, which is minimal compared to the total inference time of 0.39 sec per step" (Sec. 4.2, "Inference time").
- FLOPs: **not stated.**

## 3. Loss

**Paper form.**
- Eq. 5: $\mathcal L=\mathbb E_{t,\mathbf x_0,\mathbf x_t}[-\log q_\theta(\mathbf u\mid\mathbf x_t)]$. This is the KL to $p(\mathbf u\mid\mathbf x_t)$ with the entropy term dropped.
- Eq. 6 (Sec. 3.1), the expanded form:
$$\mathcal L=\mathbb E\Big[-\log\sum_{k=1}^K\exp\Big(-\frac{1}{2s^2}\|\mathbf u-\boldsymbol\mu_k\|^2 - D\log s + \log A_k\Big)\Big]$$
  The source comments out the constant $\frac D2\log 2\pi$, and the code omits it too.
- The norm and the $D\log s$ run over the GM's own dimension. After the pixel-wise factorization that dimension is C per pixel, summed over pixels.

**What is actually trained: the transition loss** (Sec. 3.4, "Transition loss"; Alg. `alg:train`):
$$\mathcal L_\text{trans}=\mathbb E_{t,\mathbf x_{t-\Delta t},\mathbf x_t}[-\log q_\theta(\mathbf x_{t-\Delta t}\mid\mathbf x_t)],\quad \Delta t:=\lambda t,\ \lambda\in(0,1]$$
- $q_\theta(\mathbf x_{t-\Delta t}|\mathbf x_t)$ is Eq. 9: $\sum_k A_k\mathcal N(c_1\mathbf x_t+c_2\boldsymbol\mu_{x,k},(c_3+c_2^2s_x^2)\mathbf I)$.
- Coefficients: $c_1=\frac{\sigma^2_{t-\Delta t}}{\sigma_t^2}\frac{\alpha_t}{\alpha_{t-\Delta t}}$, $c_2=\frac{\beta_{t,\Delta t}}{\sigma_t^2}\alpha_{t-\Delta t}$, $c_3=\frac{\beta_{t,\Delta t}}{\sigma_t^2}\sigma^2_{t-\Delta t}$, and $\beta_{t,\Delta t}=\sigma_t^2-\frac{\alpha_t^2}{\alpha^2_{t-\Delta t}}\sigma^2_{t-\Delta t}$ (Eq. 1).
- Paper: "When $\lambda=1$, the loss is equivalent to the original; when $\lambda<1$, the transition distribution has a lower bound of variance $c_3$, which improves training stability."
- Main model: λ = 0.5 (`configs/gmflow_imagenet_k8_8gpus.py:51`, `trans_ratio=0.5`). The K sweep (Fig. 7, Tab. 3) used an earlier run with $\lambda\sim\text{LogitNormal}(0,1)$ (App. A.4).

**Code path.**
1. Sample the times and the two noisy states (`lib/models/diffusions/gmflow.py:392-405`):
   ```python
   t_high = self.timestep_sampler(num_batches).to(device).clamp(min=eps, max=self.num_timesteps)   # eps = 1e-4
   t_low = t_high * (1 - trans_ratio)
   t_low = torch.minimum(t_low, t_high - eps).clamp(min=0)
   x_t_low, _, _ = self.sample_forward_diffusion(x_0, t_low, noise_0)
   x_t_high = self.sample_forward_transition(x_t_low, t_low, t_high, noise_1)
   ```
2. The network sees `(x_t_high, t_high)`. Its u-GM is mapped to a GM over $\mathbf x_{t_\text{low}}$ (`gmflow.py:185-204`):
   ```python
   c1 = sigma_to_over_sigma ** 2 * alpha_over_alpha_to
   c2 = beta_over_sigma_sq * alpha_to
   c3 = beta_over_sigma_sq * sigma_to ** 2
   means_x_0 = x_t_high - sigma * denoising_output['means']
   logstds_x_t_low = torch.logaddexp(
       (denoising_output['logstds'] + torch.log((sigma * c2).clamp(min=eps))) * 2,
       torch.log(c3.clamp(min=eps))) / 2
   means_x_t_low = c1 * x_t_high + c2 * means_x_0
   ```
3. GM NLL with target `x_t_low` (`lib/models/losses/diffusion_loss.py:33-36`):
   ```python
   inverse_std = torch.exp(-pred_logstds).clamp(max=1 / eps)
   diff_weighted = (pred_means - target.unsqueeze(-4)) * inverse_std
   gaussian_ll = (-0.5 * diff_weighted.square() - pred_logstds).sum(dim=-3)  # sum over C channels
   loss = -torch.logsumexp(gaussian_ll + pred_logweights.squeeze(-3), dim=-3)  # over K -> (bs, h, w)
   ```

**Reduction.**
- The loss is **summed over the C channels** inside each pixel and **averaged over pixels** (`reduction='flatmean'`, `diffusion_loss.py:313`). mmgen's `flatmean` is `loss.mean(dim=list(range(1, loss.ndim)))`.
- It is then **averaged over the batch** (`reduce_loss(..., 'mean')`, `:363`).
- It is then **multiplied by 2** (`weight_scale=2.0`, config `:38`; `loss_rescaled = loss * self.weight_scale`, `:344`).
- Per sample the loss is therefore $\frac{2}{HW}\sum_{\text{pixels}}\text{NLL}_\text{pixel}$. It is not normalized by C.

**Other loss details.**
- **Per-timestep weighting:** none. `GMFlowNLLLoss.forward` multiplies only by `weight_scale` (`:344`).
- **Detached gradients:** the s-MLP input is `cond_emb.detach()` (`gmflow.py:403`), so the variance branch does not train the time/class embedder. Inside the loss, nothing is detached: μ, logits and log s all receive gradients.
- **Auxiliary L2 term on the mean:** none.
- **Spectral loss:** the only extra loss (App. A.3, label `eq:specloss`), with weight 1.0. It trains only the spectrum MLP. Its inputs are built under `torch.no_grad()` (`gmflow.py:369-375`), and the paper says "we stop the gradient flow through KR transport to prevent spectral learning from influencing the main GMFlow model." Code: `loss = z_kr_fft.square() * (torch.exp(-log_var) - 1) + log_var; loss = loss.mean() * (0.5 * num_channels * self.spectral_loss_weight)` (`:379-380`).
- **Timestep distribution:** logit-normal(0, 1) with shift 1.0 (App. A.4: "Following Stable Diffusion 3 ... sample $t$ from a logit-normal distribution"). Config `:40`. In `lib/models/diffusions/sampler.py:25-31`: `t_linear = torch.sigmoid(mean + std * torch.randn(...)); t = shift * t_linear / (1 + (shift - 1) * t_linear); return t * num_timesteps`.

**Numerics.**
- The loss and the transition math run in fp32 with autocast off (`gmflow.py:347-351`, `with torch.autocast(device_type='cuda', enabled=False): x_t_low = x_t_low.float() ...`).
- The network runs under bf16 autocast (config `:43`).
- The loss uses `logsumexp` over K, `logaddexp` for the variance sum, `clamp(min=1e-6)` on the logs of $c_2$, $c_3$ and σ, and the inverse-std clamp at 1e4.
- **Non-finite gradient guard:** gradient clipping at 10 from iteration 1000 uses `error_if_nonfinite=True`. On a NaN or Inf gradient the step is skipped and the grads are zeroed (`lib/models/base.py:12-21`; config `:52-54`).
- **Optimizer:** AdamW8bit, lr 2e-4, betas (0.9, 0.95), wd 0.05, linear warmup over 1000 iterations (config `:60-67, 82-86`). Batch 4096, 200K iterations (App. A.4). Class dropout 10% (`prob_class=0.9`, config `:52`). EMA on.

**Why the variance needed help (paper).**
- Sec. 3.1: "The expansion of Eq. (5) involves the inverse covariance matrix $\boldsymbol\Sigma_k^{-1}$, which can lead to training instability."
- Sec. 3.4: "the GM KL loss may still induce minor gradient spikes when $s$ becomes small."
- The source also has a commented-out author note after Eq. 6: `% \fujun{Here, when $s$ is small, numerical issues.}`

## 4. K = 1 with learned variance: gradient on the mean

**What the sources say.**
- Paper, Sec. 3.1, "Special cases": "In a special case where $K=1, s=1$, Eq. (6) is identical to the $L_2$ loss in Eq. (3)." The paper states the identity only for **fixed s = 1**.
- Paper, Sec. 4.1: "GM-SDE sampling with K = 1 is theoretically equivalent to DDPM with learned variance."
- Neither source discusses whether a learned s reweights timesteps: **not stated.**
- The code has no special K = 1 path. It also has no stop-gradient that would make the mean see plain L2.

**My derivation (inference, not in the sources).**
- Plain KL loss (λ = 1), K = 1: $\mathcal L=\frac{\|\mathbf u-\boldsymbol\mu\|^2}{2s^2}+D\log s$, so $\nabla_{\boldsymbol\mu}\mathcal L=(\boldsymbol\mu-\mathbf u)/s^2$.
- s depends only on (t, c) and is shared over pixels. The mean therefore gets the L2 flow-matching gradient **reweighted per sample by $1/s(t,c)^2$**.
- At the optimum for s, $s^2(t,c)$ is the per-dimension residual MSE at that (t, c). The effective weight is then about $1/\text{MSE}(t)$, which equalizes the per-timestep loss scale. So the gradient is **not** the same as standard L2.
- Transition loss, λ < 1, K = 1: $\mathbf x_{t-\Delta t}-\text{mean}=c_2\sigma_t(\boldsymbol\mu-\mathbf u)+\sqrt{c_3}\,\xi$. The weight on $(\boldsymbol\mu-\mathbf u)$ becomes $w(t)=\frac{(c_2\sigma_t)^2}{(c_2\sigma_t)^2s^2+c_3}$.
- With λ = 0.5 and the linear schedule, this weight is about $1/s^2$ for large s. As $s\to0$ it is **capped at $3-2t$**: $(c_2\sigma_t)^2/c_3 = (2-t)^2-(1-t)^2$. So it lies between 1 and 3 when s → 0.
- This is the mechanism behind "lower bound of variance $c_3$": the transition loss bounds the reweighting that a small learned s would otherwise blow up.

**Reported K = 1 and L2-baseline numbers.**
- Tab. 3 (validation NLL of ImageNet latents, bits/dim, $t\sim U(0,1)$, λ = 1 form, earlier random-λ training): **K=1 0.346; K=4 0.263; K=8 0.242; K=16 0.224; K=16 + spectral 0.173.**
- Fig. 7 (`plot_k_sweep.pdf`) plots best FID and best Precision against K for GM-SDE 2 and GM-ODE 2 at NFE = 8. The values appear **only as plotted points**; the text gives no number. Text: "Few-step sampling (NFE = 8) benefits the most from increasing GM components, particularly with the GM-SDE 2 solver. The metrics generally saturate at K = 8. Beyond this point, the GM-SDE 2 Precisions decline because spectral sampling introduces larger numerical errors when more Gaussian components are used."
- Fig. 3 (checkerboard): K = 1 GM-SDE "produces a blurrier histogram". This is a qualitative result only.
- L2 baseline against GMFlow K = 8 on ImageNet: see Tab. 2 and Tab. 4 in Sec. 8. The baseline is the same DiT-XL/2 trained with Eq. 3 and run with standard solvers.
- **No ImageNet run isolates "K = 1 + learned s" against "L2" with the same solver.**

## 5. GM-SDE sampler

**From the velocity GM to the denoising GM.** Substitute $\mathbf u=(\mathbf x_t-\mathbf x_0)/\sigma_t$ (Sec. 3.1): $q_\theta(\mathbf x_0|\mathbf x_t)=\sum_kA_k\mathcal N(\mathbf x_0;\mathbf x_t-\sigma_t\boldsymbol\mu_k,(\sigma_ts)^2\mathbf I)$. Code: `gmflow.py:105-106` (quoted in Sec. 1).

**Update** (Sec. 3.3, Eq. 9 and the two-stage form):
"first sampling $\hat{\mathbf x}_0\sim q_\theta(\mathbf x_0|\mathbf x_t)$ and then sampling $p(\mathbf x_{t-\Delta t}|\mathbf x_t,\hat{\mathbf x}_0)=\mathcal N(\mathbf x_{t-\Delta t};c_1\mathbf x_t+c_2\hat{\mathbf x}_0,c_3\mathbf I)$". The second factor is the DDPM/DDIM bridge posterior. Its derivation is in App. C.3, label `eq:transbayes`.

Code, `lib/models/diffusions/schedulers/gmflow_sde.py:126-145`:
```python
sigma_to_over_sigma = sigma_to / sigma.clamp(min=eps)
alpha_over_alpha_to = alpha / alpha_to.clamp(min=eps)
beta_over_sigma_sq = 1 - (sigma_to_over_sigma * alpha_over_alpha_to) ** 2
c1 = sigma_to_over_sigma ** 2 * alpha_over_alpha_to
c2 = beta_over_sigma_sq * alpha_to
if self.config.output_mode == 'sample' or self.config.variance_mode == 'small':
    std = beta_over_sigma_sq ** 0.5 * sigma_to  # c3_sqrt
prev_sample = c1 * sample + c2 * x0 + std * noise
```

**Drawing $\hat{\mathbf x}_0$.** There are two modes.
1. **Independent per pixel.** For each pixel, pick a component index by multinomial over the K weights, then add isotropic Gaussian noise with std $s_x$ to all C channels. In `lib/ops/gmflow_ops/gmflow_ops.py:637-655`:
   ```python
   inds = torch.multinomial((gm_logweights...reshape(batch_numel * h * w, num_gaussians) * gm_power).softmax(dim=-1), n_samples, replacement=True)
   samples = stds * torch.randn(...) + means
   ```
2. **Spectral sampling (default in GM-SDE when the spectrum net exists).** This mode correlates pixels (Sec. 3.4; App. A.3).
   - Draw a frequency-domain Gaussian with a predicted power spectrum and build a Hermitian array.
   - Apply `irfft2(norm='ortho')` to get a spatially correlated standard-Gaussian field.
   - Map each pixel's Gaussian sample to a sample of that pixel's GM by Knothe–Rosenblatt transport. This is a per-pixel conditional 1-D GM inverse CDF along the eigenaxes of the image-averaged covariance (`axis_aligned=True`).
   - Code: `gmflow.py:226-236`:
     ```python
     spectral_samples = torch.randn(shape, ...) * (power_spectrum / 2).exp()
     z_kr = torch.complex(z_0 + z_1, z_0 - z_1) / 2
     gaussian_samples = torch.fft.irfft2(z_kr, norm='ortho')
     samples = gaussian_samples_to_gm_samples(gm, gaussian_samples, axis_aligned=True)
     ```
     The transport itself is `gmflow_ops.py:990-1117` (inverse-CDF solve, `n_steps=16`).
   - Spectrum MLP inputs: "the mean of per-pixel GM variances and the variance of per-pixel GM means". Its output is a log-power spectrum, normalized by a softmax over frequencies (`gmflow.py:731-754`, `log_var = spectrum.flatten(-2).log_softmax(dim=-1) + math.log(h * w)`).
- **Temperature:** none in GMFlow. pi-Flow's temperature is described in Sec. 1.

**Step schedule.**
- Uniform in σ: `sigmas = 1 - linspace(0, 1, NFE, endpoint=False)`, then a shift map (shift = 1.0 here, so no change), with 0 appended (`gmflow_sde.py:65-69`).
- Alg. `alg:sample`: $t\gets1$, $\Delta t\gets 1/\mathit{NFE}$.
- NFE reported: 4, 6, 8, 32 (Fig. 6 sweeps NFE). Ablations use 8. Tab. 2 uses 32. The test config sweeps NFE ∈ {8, 32} (`configs/gmflow_imagenet_k8_test.py:73`).

**Last step.** At `sigma_to = 0` the coefficients become $c_1=0$, $c_2=1$ and std $=0$. The output is the drawn sample $\hat{\mathbf x}_0$, not the GM mean. The paper and the code give no special final-step rule beyond this.

**Second-order variant ("GM-SDE 2", App. A.2, label `eq:gm2`).**
1. Convert the previous step's GM to the current $\mathbf x_t$ with Eq. 10. This gives $\hat q(\mathbf x_0|\mathbf x_t)$.
2. Fit isotropic surrogates $\mathcal N(\boldsymbol\mu_+,s_+^2)$ to the current GM and $\mathcal N(\boldsymbol\mu_-,s_-^2)$ to $\hat q$.
3. Reweight the current GM by
$$\rho(\mathbf x_0)=\frac{\mathcal N(\mathbf x_0;\boldsymbol\mu_++\Delta\boldsymbol\mu,(s_+^2-\|\Delta\boldsymbol\mu\|^2/D)\mathbf I)}{\mathcal N(\mathbf x_0;\boldsymbol\mu_+,s_+^2\mathbf I)},\qquad \Delta\boldsymbol\mu=\tfrac{\boldsymbol\mu_+-\boldsymbol\mu_-}{2}$$
4. Rescale $\Delta\boldsymbol\mu$ by $\sqrt{\max(0,1-\frac{(\tilde w^2+c_a)s_c^2}{\Delta t^2})}$, with $c_a=0.005$.

Code: `gmflow.py:271-291`:
```python
k = 0.5 * h / self.prev_h
err_power = avg_var * (_guidance_scale * _guidance_scale + ca)
mean_diff = (gm_mean - mean_from_prev) * ((1 - err_power / (prev_h_norm * prev_h_norm)).clamp(min=0).sqrt() * k)
bias = bias * (avg_var / bias_power.clamp(min=1e-6)).clamp(max=1).sqrt()
```
- The bias power is capped at the average variance (`:284`).
- The first step falls back to first order (`:312-313`).

## 6. GM-ODE sampler and analytic sub-steps

**Paper (Sec. 3.3, Eq. 10):** $\hat q(\mathbf x_0|\mathbf x_\tau)=\frac{p(\mathbf x_\tau|\mathbf x_0)}{Z\,p(\mathbf x_t|\mathbf x_0)}q_\theta(\mathbf x_0|\mathbf x_t)$ for $\tau<t$. The derivation is in App. C.5. The result is again a GM, because the ratio of two Gaussians in $\mathbf x_0$ is a Gaussian (App. D.1–D.2).
- Sub-step (Alg. `alg:sample`, lines 18–22): $\mathbf x_{\tau-h}\gets\mathbf x_\tau-h\,\mathbb E_{\mathbf x_0\sim\hat q(\mathbf x_0|\mathbf x_\tau)}\big[\frac{\mathbf x_\tau-\mathbf x_0}{\sigma_\tau}\big]$, with $h=\Delta t/n$.

**Closed form** (pi-Flow App. "Derivation of Closed-Form GMFlow Velocity"; same math as the code). With $\boldsymbol\nu=\frac{\alpha_\tau\mathbf x_\tau}{\sigma_\tau^2}-\frac{\alpha_t\mathbf x_t}{\sigma_t^2}$ and $\zeta=\frac{\alpha_\tau^2}{\sigma_\tau^2}-\frac{\alpha_t^2}{\sigma_t^2}$:
$${s'}^2=\frac{s_x^2}{s_x^2\zeta+1},\quad \boldsymbol\mu'_k=\frac{s_x^2\boldsymbol\nu+\boldsymbol\mu_{x,k}}{s_x^2\zeta+1},\quad a'_k=\log A_k+\frac{\boldsymbol\mu_{x,k}\cdot(\boldsymbol\nu-\frac12\zeta\boldsymbol\mu_{x,k})}{s_x^2\zeta+1}$$
The velocity is $(\mathbf x_\tau-\sum_kA'_k\boldsymbol\mu'_k)/\tau$.

Code, `lib/models/diffusions/gmflow.py:54-70`:
```python
denom = (alpha_tgt_sigma_src.square() - alpha_src_sigma_tgt.square()).clamp(min=eps)
g_mean = (alpha_tgt_sigma_src * sigma_src * x_t_tgt - alpha_src_sigma_tgt * sigma_tgt * x_t_src) / denom
g_var = (sigma_tgt * sigma_src).square() / denom
out_means = (g_var * gm_means + gm_vars * g_mean) / norm_factor
logweights_delta = gm_diffs.square().sum(dim=-3, keepdim=True) * (-0.5 / norm_factor)
out_weights = (gm_logweights + logweights_delta).softmax(dim=-4)
```
- The sub-step loop is `gmflow.py:527-539`.
- Sub-step 0 uses the GM mean directly. Later sub-steps call `denoising_gm_convert_to_mean`.
- Each sub-step is a plain Euler step, `FlowEulerODEScheduler.step`. In `flow_euler_ode.py:119,122`: `derivative = (sample - model_output) / sigma; prev_sample = sample + derivative * dt`.
- The sub-step grid is the uniform σ grid of `NFE·n` points (`gmflow.py:456`, `sampler.set_timesteps(num_timesteps * num_substeps)`).
- At the last sub-step σ → 0, and the output is the GM mean.

**Number of sub-steps** (App. A.4): "$n=\max(\frac{128}{\mathit{NFE}},2)$, which performs well when NFE ≥ 8. For the exception when NFE = 4 or 6, we observe that reducing $n$ to 8 yields better performance."
- Test config: (NFE, n) = (8, 16) and (32, 4) (`gmflow_imagenet_k8_test.py:73`).
- Training-time eval: 16 steps × 8 sub-steps (`gmflow_imagenet_k8_8gpus.py:94-95`).
- README demo: 32 × 4.
- Checkerboard: $n=\lceil128/\mathit{NFE}\rceil$ (Sec. 4.1).

**GM-ODE 2** uses the same second-order GM extrapolation as GM-SDE 2 and then the sub-steps.

**When ODE and when SDE.** The paper gives **no selection rule: not stated.** It reports both. At NFE = 8 (Tab. 4), GM-ODE 2 has the better best FID (2.77 vs 3.43 for GM-SDE 2) and the better best Precision (0.946 vs 0.939). At NFE = 32 (Tab. 2), GM-SDE reaches 0.950 Precision against 0.947 for GM-ODE. The paper also says GM-ODE 2 "reaches a Precision of 0.942 in just 6 steps".

## 7. Probabilistic guidance (one paragraph)

Sec. 3.2 (Eqs. 7–8) multiplies the conditional GM by a Gaussian ratio. The ratio shifts the conditional surrogate mean by $\tilde w s_c\Delta\boldsymbol\mu_n$ and shrinks its variance by $(1-\tilde w^2)$, where $\tilde w\in[0,1)$. This keeps the per-dimension spread and needs an **unconditional** prediction (class dropout 10% in training). **No sampler requires guidance.**
- The code enables guidance only when `guidance_scale > 0.0` (`gmflow.py:463`, `use_guidance = guidance_scale > 0.0`). With w̃ = 0 the GM goes straight to the solver (`:500-502`).
- The second-order solver has a no-guidance branch (`:259-264`). Its damping term then keeps only $c_a$ (`err_power = avg_var * (0 + ca)`).
- Spectral sampling uses the GM statistics or the guided surrogate, whichever exists (App. A.3 caption).
- Caveat: **every ImageNet number in the paper uses guidance** (swept w̃ ∈ {0.02 … 0.75}, App. A.4) plus orthogonal projection (`orthogonal_guidance=1.0`). There is **no unguided ImageNet number**.
- The only unguided results are unconditional: the checkerboard (Figs. 2–3) and CIFAR-10 (Tab. 5).

## 8. Ablations

**Tab. 4 (ImageNet, K = 8, NFE = 8; best FID ↓ / best Precision ↑ / Saturation at best Precision):**

| ID | Setting | Best FID | Best Prec. | Sat. |
|---|---|---|---|---|
| A0 | Full model (GM-SDE 2) | 3.43 | 0.939 | −0.003 |
| A1 | A0 → GM-SDE (1st order) | 6.96 | 0.938 | +0.009 |
| A2 | A1 w/o spectral sampling | 8.98 | 0.940 | −0.022 |
| A3 | A2 → vanilla CFG | 9.02 | 0.917 | +0.049 |
| A4 | A0 → DPM++ 2M SDE | 4.59 | 0.912 | −0.062 |
| A5 | A0 → λ = 1.0 (plain KL loss) | 4.49 | 0.941 | −0.019 |
| B0 | Full model (GM-ODE 2) | 2.77 | 0.946 | −0.028 |
| B1 | B0 w/o sub-steps | 7.47 | 0.947 | −0.031 |

**Tab. 2 (ImageNet, best Precision, NFE = 32; Saturation relative to real data 0.318):**
- With orthogonal projection:

  | Solver | Guidance | Precision | Saturation |
  |---|---|---|---|
  | DDPM small | w = 3.3 | 0.944 | +0.032 |
  | DPM++ 2M SDE | w = 2.9 | 0.938 | +0.032 |
  | GM-SDE | w̃ = 0.47 | 0.950 | −0.019 |
  | GM-SDE 2 | w̃ = 0.39 | 0.950 | −0.019 |
  | Euler | w = 3.3 | 0.934 | +0.052 |
  | UniPC | w = 3.3 | 0.931 | +0.048 |
  | GM-ODE | w̃ = 0.47 | 0.947 | −0.024 |
  | GM-ODE 2 | w̃ = 0.47 | 0.947 | −0.024 |

- Without orthogonal projection:

  | Solver | Guidance | Precision | Saturation |
  |---|---|---|---|
  | DDPM small | w = 3.3 | 0.939 | +0.056 |
  | DPM++ 2M SDE | w = 2.9 | 0.933 | +0.054 |
  | GM-SDE | w̃ = 0.27 | 0.943 | +0.023 |
  | Euler | w = 3.3 | 0.931 | +0.064 |
  | UniPC | w = 3.3 | 0.929 | +0.060 |
  | GM-ODE | w̃ = 0.27 | 0.933 | +0.016 |

**K.**
- Tab. 3 NLL: see Sec. 4.
- Fig. 7: values are plotted only. The text says the metrics saturate at K = 8 and GM-SDE 2 Precision falls beyond K = 8.
- Checkerboard (Fig. 3): K ∈ {1, 2, 4, 8, …, 64} shown qualitatively. "With K = 8, the GM probabilities are sufficiently accurate that the second-order solvers do not make a difference."
- pi-Flow, Tab. `tab:imagenet_abl` (a distilled 1-NFE student, not GMFlow training): GM K = 8 FID 3.07 against K = 32 FID 3.08 (REPA teacher).

**Variance learned vs. fixed.** **No ablation in either source.**
- The baselines "DDPM small/large" are GM-SDE with K = 1 and fixed $s=0$ or $s=1/\sqrt{\alpha_t^2+\sigma_t^2}$, run on the **L2 model** (App. A.5). They are sampler baselines, not a training ablation.

**Sampler vs. NFE.** Fig. 6(a) plots best FID and best Precision against NFE for GM-SDE/ODE 2 and the baselines (DDPM, Euler, DPM++, DEIS, UniPC, SA-Solver). The values are plotted only. The text says GMFlow wins FID at NFE ≤ 8 and stays "competitive" at NFE ≥ 32.

**CIFAR-10, unconditional, SDE sampling, FID (Tab. 5; K = 2, U-Net):**

| NFE | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| DDPM large | 205.31 | 84.71 | 37.35 | 14.81 |
| DDPM small | 34.76 | 16.18 | 11.11 | 8.38 |
| SN-DDPM | 16.33 | 6.05 | 4.19 | 3.83 |
| GMS | 13.80 | 5.48 | 4.00 | 3.46 |
| GM-SDE 2 | 9.11 | 4.16 | 3.79 | 3.76 |

**Precision and recall.**
- Precision is reported throughout.
- Recall appears only as Precision–Recall curves at NFE = 8 (Fig. 6(b)), with no table. The text says GMFlow "achieves superior Precision and Recall".
- App. A.4: the random-λ setting "slightly increases Precision and decreases Recall".

## 9. Collapsing or dead components; spread of the initial means

- **Paper, Sec. 3.4:** "For high-dimensional data such as images, Gaussian mixture models often suffer from mode collapse. To address this, we treat each pixel ... as an independent low-dimensional GM." This is the only statement about collapse. It concerns the dimension of each GM, not individual components.
- **Dead or unused components, load balancing, entropy bonus, per-component variance: not stated** in the paper or in the code. The code has no auxiliary term on the weights.
- **Spread of the initial means: not stated** in either source. As built, the released head Xavier-initializes the means and the logits together (`gmflow.py:198, 296-297`), with only `proj_out_1` zeroed. The initial means are therefore distinct random projections, and the initial weights are not uniform (Sec. 1).
- **Large K:** "GM-SDE 2 Precisions decline because spectral sampling introduces larger numerical errors when more Gaussian components are used" (Sec. 4.2).
- **pi-Flow App. "Discussion on GMFlow Policy Hyperparameters":** "A larger K improves expresiveness but impairs compatibility as it may complicate network training. A larger C improves robustness (since GMFlow models correlations within each C-dimensional chunk) but impairs expresiveness."
- **pi-Flow factorization toy test** (K = 32, fitted to FLUX): "element-wise factorization (C = 1) leads to over-saturated colors, high contrast, and over-smoothed textures, while patch-wise factorization (C = 2 × 2 × VAE latent channels) results in 'confetti' artifacts and noisy textures."
- **pi-Flow "GM dropout":** in distillation it masks components "synchronously across all i" and renormalizes. This diversifies rollouts: FID 4.14 → 3.65, K = 32, FM teacher. It is a distillation trick, not a GMFlow training trick.

## 10. GitHub issues

I fetched all issues and PRs through `api.github.com/repos/Lakonik/GMFlow/issues?state=all` on 2026-11-05. **There are 3 issues, all closed, and 0 PRs.** None reports NaN, instability, collapsing components, small K, pixel space, variance or sampler problems.
- **#1 "Paper Typo"**, https://github.com/Lakonik/GMFlow/issues/1. Relevant to guidance only. The author (Lakonik) wrote: "0.4 is the scale for best Precision, whereas 0.04 is the scale for best FID." On large scales: "A very large guidance scale (e.g., w>0.7) would still cause oversaturation, since the guidance bias may accumulate along the same direction during the sampling iterations. Still, if your implementation is correct (which requires a flow matching model with at least one Gaussian component plus learned variance), then probabilistic guidance should significantly reduce oversaturation compared to standard CFG."
- **#2 "Code running error"**, https://github.com/Lakonik/GMFlow/issues/2. Dependency conflict (`KeyError: 'SiLU is already registered in activation layer'`), no reply. Not relevant.
- **#3 "Pretrained Checkpoint for Imagenet"**, https://github.com/Lakonik/GMFlow/issues/3. Checkpoint at `huggingface.co/Lakonik/gmflow_imagenet_k8_ema`. Not relevant.

## 11. License

**MIT License, "Copyright (c) 2025 Hansheng Chen"** (`LICENSE`). Source files carry `# Copyright (c) 2025 Hansheng Chen`. The code depends on mmcv, mmgen and diffusers. I did not check those licenses.

---

## 12. Port notes for a pixel-space x-prediction model

**Everything in this section is my inference. Neither source states it.**

1. **Time mapping.** Ours: $\mathbf x_t=t\mathbf x+(1-t)\epsilon$, with data at t = 1. GMFlow: $\sigma=1-t_\text{ours}$ and $\alpha=t_\text{ours}$. GMFlow's $\mathbf u=(\mathbf x_t-\mathbf x)/(1-t)=-\mathbf v_\text{ours}$. All $c_1,c_2,c_3$ formulas carry over with $\sigma\to1-t$ and $\alpha\to t$.

2. **Keep x-prediction for the means.** Let the head output K x-means $\boldsymbol\mu_{x,k}$ per pixel, K logits per pixel, and one $\log s_x$ per sample.
   - The u-parameters, which the solvers need, are then $\boldsymbol\mu_k=(\mathbf x_t-\boldsymbol\mu_{x,k})/(1-t)$ and $s_u=s_x/(1-t)$.
   - Code `u_to_x_0` (`gmflow.py:93-120`) is this map read backwards. The solvers internally work in x0 form anyway (`prediction_type='x0'`).

3. **The NLL makes the x-loss and the v-loss equivalent.** $q_x(\mathbf x_0)=\sigma^{-D}q_u(\mathbf u)$, so $-\log q_u=-\log q_x-D\log\sigma$. The two losses differ by a constant that does not depend on θ, so the **gradients are identical**.
   - This differs from L2, where the v-loss is the x-loss times $1/(1-t)^2$.
   - The learned $s_x(t)$ sets the time weighting instead. The weight is $1/s_x^2$ for K = 1 (Sec. 4), and with the transition loss it is bounded as in Sec. 4.
   - Consequence: the JiT clamp $1-t\ge0.05$ is not needed in the loss. It appears only where the sampler converts to velocity ($/\sigma$). `gmflow.py` clamps σ at 1e-6 there.

4. **The weighting change is a confound.** Moving from our v-L2 loss to a GM NLL changes two things at once: the distribution, and the effective per-t weight (from $1/(1-t)^2$ to about $1/\text{MSE}_x(t)$, capped by the transition loss).
   - Control to isolate the mixture: a **K = 1, learned-s** run with the same transition loss.
   - A fixed-s run reproduces the plain L2 weighting only if $s_x\propto(1-t)$ (that is, $s_u$ constant). Code `constant_logstd` fixes $s_u$, which corresponds to this case.

5. **RGB instead of latent channels.**
   - Per pixel, C = 3. This is close to pi-Flow's "element-wise (C = 1)" case, which gave over-saturated, over-smoothed images in its toy test. Latent C = 4 or 16 behaved best.
   - The variance has no channel structure (isotropic). The mixture over K is the only within-pixel RGB correlation.
   - Alternative granularity: one GM per patch with C = p²·3. The paper never tried this. pi-Flow's patch-wise test (C = 2×2×latent) gave "confetti" artifacts.
   - A middle option, such as one GM per 2×2 RGB block (C = 12), is untested in both sources.

6. **Head size for JiT.** Per token, the final linear emits $p^2\cdot K\cdot(3+1)$ values.
   - p = 16, K = 8: 8192 outputs instead of 768. At width 640 that is about 5.2 M params instead of 0.49 M. Compute is about 2·640·8192 FLOP per token, still small next to the trunk.
   - The loss tensor $(B,K,3,H,W)$ at 256² and K = 8 is about 1.6 M floats per image, in fp32 for the loss.
   - Keep the loss in fp32 with `logsumexp`, as `gmflow.py:347-351` does.

7. **The variance head.** Copy the design: a scalar $\log s$ per sample from a small MLP on the **detached** time embedding plus the condition embedding. For text-to-image this would be the pooled text embedding; the paper used the class embedding.
   - Predict $\log s_x$ directly, or predict $\log s_u$ and add $\log(1-t)$.
   - $s_x\to0$ as $t\to1$. A floor on $\log s_x$, or the λ < 1 transition loss, is the paper's remedy for the "gradient spikes when s becomes small".
   - The released code has only the inverse-std clamp at 1e4 and the non-finite-gradient step skip.

8. **Initialization.**
   - Zero-init the logit and $\log s$ outputs (V2 / toy style), so the weights start uniform and s = 1. In x-form: $s_x=(1-t)$ at init if $s_u$ is predicted.
   - Do not zero the K mean outputs. Identical means at init give identical gradients through the responsibilities only up to the logit noise. The released model keeps the means Xavier-random. The paper is silent on component death, so monitor the per-pixel weight entropy and the per-component usage.

9. **Sampling without CFG.**
   - Set w̃ = 0. The GM-ODE (2) and GM-SDE (2) code paths run unchanged (Sec. 7).
   - GM-ODE uses the full distribution through the sub-step transfer (Eq. 10). It returns the GM mean at the end, so it never draws per-pixel independent samples. It is the safer first test in pixel space.
   - GM-SDE without spectral sampling draws $256^2$ independent per-pixel samples at each step, including the final step, which returns a sample. In pixel space this adds uncorrelated per-pixel noise. The paper's own ablation shows independent sampling costs FID: A1 → A2, 6.96 → 8.98.
   - Spectral sampling ports directly (FFT over H×W with C = 3), but its MLP is tied to a fixed base size (`SpectrumMLP(base_size=(C, H, W))`).

10. **Transition loss in our convention.**
    - Draw $t_\text{high}$ on the GMFlow σ axis, then $t_\text{low}=(1-\lambda)t_\text{high}$ with λ = 0.5.
    - Build $\mathbf x_{t_\text{low}}$ from $\mathbf x$, and $\mathbf x_{t_\text{high}}$ by the forward transition (`gmflow.py:77-91`).
    - Score $\mathbf x_{t_\text{low}}$ under $\sum_kA_k\mathcal N(c_1\mathbf x_{t_\text{high}}+c_2\boldsymbol\mu_{x,k},(c_2^2s_x^2+c_3)\mathbf I)$.
    - With x-means, $c_2$ multiplies $\boldsymbol\mu_{x,k}$ directly, with no σ factor. The code's `prediction_type='x0'` branch (`gmflow.py:196-201`) is exactly this case.
