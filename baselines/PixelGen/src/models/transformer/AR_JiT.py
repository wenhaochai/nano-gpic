"""An image model that works as an LLM does (docs/autoregressive.md): at each step it gives the distribution of the next patch given the caption and the patches before it, and the sampler draws from it.

p(x | y) = prod_i p(patch_i | patch_1 ... patch_{i-1}, y). Two orders. order = "raster": left to right, with a causal mask, as an LLM: z_i sees the caption and the patches before i. order = "random": a new random order per image and bidirectional attention, as MAR (Li et al. 2024; its table 1 has FID 19.23 for raster and causal against 3.43 for random and bidirectional without CFG): the known patches and a mask token at the unknown positions all attend to each other. Unlike MAR, one stack runs over all positions, with no split into an encoder and a decoder.
The backbone is JiT_T2I's. There is no noise level, and the backbone's timestep embedding sees a constant.
The distribution of one patch is continuous: a small flow on the 768 values of the patch, given z_i (TokenFlowHead), with the official recipe's x-prediction and v-space L2.
"""
import math
import os

import torch
import torch.nn as nn

from src.models.layers.time_embed import TimestepEmbedder
from src.models.transformer.JiT_T2I import JiT_T2I, ResBlock


class TokenFlowHead(nn.Module):
    """A flow on one patch given z: x_hat = head(q, tau, z), with q = tau patch + (1 - tau) eta."""

    def __init__(self, patch_dim, cond_dim, width=1024, depth=3):
        super().__init__()
        self.proj_in = nn.Linear(patch_dim, width)
        self.proj_cond = nn.Linear(cond_dim, width)
        self.t_embedder = TimestepEmbedder(width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(width, patch_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for blk in self.blocks:   # each block starts as the identity
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out.weight, 0)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, q, tau, z):
        """q [M, P], tau [M], z [M, D]."""
        c = self.t_embedder(tau) + self.proj_cond(z)
        h = self.proj_in(q)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(self.norm(h))


class HeadFlow(nn.Module):
    """The head's flow as one module, so that torch.compile sees the whole loop: one graph per draw, not one Python call per flow step. Same math as draw_patch's loop. At evaluation the GPUs sat at 54 to 56% with one Python thread at 100% per GPU (w1151)."""

    def __init__(self, head, steps, t_eps):
        super().__init__()
        self.head, self.steps, self.t_eps = head, steps, t_eps

    def forward(self, q, z):
        B = z.shape[0]
        grid = torch.linspace(0, 1, self.steps + 1, device=z.device)
        for j in range(self.steps):
            hat = self.head(q, grid[j].repeat(B), z).float()
            q = hat if j == self.steps - 1 else q + (grid[j + 1] - grid[j]) * (hat - q) / (1 - grid[j]).clamp_min(self.t_eps)
        return q


class ARJiT(JiT_T2I):
    is_ar = True

    def __init__(self, head_width=1024, head_depth=3, head_steps=32, t_eps=0.05, context_noise=0.0, order="raster", ar_steps=64, temperature=1.0, eta=0.0, redraw_first=0, **kwargs):
        """context_noise: Gaussian noise of this std on the patches that the backbone reads, in training and at sampling, so that a drawn patch with small errors is like the inputs it trained on (the conditioning augmentation of Cascaded Diffusion). Targets stay clean. 0 is the first design."""
        super().__init__(**kwargs)
        self.patch_dim = self.in_channels * self.patch_size ** 2
        self.head_steps, self.t_eps, self.context_noise, self.temperature = head_steps, t_eps, context_noise, temperature   # temperature: the scale of the head's start noise on its ODE. Not MAR's temperature, which scales the noise of each reverse step
        self.z_norm = self.final_layer.norm_final
        del self.final_layer   # the output is the head's; an unused parameter would stop DDP
        assert order in ("raster", "random"), order
        self.order, self.ar_steps, self.eta, self.redraw_first = order, ar_steps, eta, redraw_first   # eta and redraw_first choose the sampler of ar_sample, in random order
        if order == "raster":
            self.bos = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
            self.register_buffer("causal", torch.tril(torch.ones(self.num_patches, self.num_patches, dtype=torch.bool)), persistent=False)
        else:
            self.mask_token = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        self.head = TokenFlowHead(self.patch_dim, self.hidden_size, head_width, head_depth)

    def forward(self, tokens, y, y_mask=None, q=None, tau=None, known=None):
        """tokens [B, N, P]: the patches in raster order. Raster order: the output at i sees tokens before i only, whatever tokens[:, i:] holds. Random order: known [B, N] says which tokens the model sees, and the others do not reach any output.
        Without q: z [B, N, D]. With q [B, N, P] and tau [B, N]: the head's x_hat [B, N, P], inside this graph (training)."""
        B, N, P = tokens.shape
        side = int(N ** 0.5)
        xpos = self.fetch_pos(side, side, tokens.device)
        ypos = self.y_pos_embedding if y_mask is None else self.y_pos_embedding[:, :y.shape[1]]
        t = self.t_embedder(torch.zeros(B, device=tokens.device, dtype=tokens.dtype)).view(B, -1, self.hidden_size)
        y = self.y_embedder(y).view(B, -1, self.hidden_size) + ypos.to(y.dtype)
        condition = nn.functional.silu(t)
        text_mask = None if y_mask is None else y_mask[:, None, None, :]
        for block in self.text_refine_blocks:
            y = block(y, condition) if text_mask is None else block(y, condition, text_mask)
        if self.order == "raster":
            x = self.x_embedder(tokens[:, :-1])
            x = torch.cat([self.bos.expand(B, 1, self.hidden_size).to(x.dtype), x], dim=1) + self.pos_embed.to(x.dtype)   # bf16 under autocast, as the official in-place add keeps it (review 2026-11-04)
            causal = self.causal[None, None].expand(B, 1, N, N)   # SDPA boolean mask, True = attend
            words = torch.ones(B, 1, N, y.shape[1], dtype=torch.bool, device=x.device) if y_mask is None else y_mask[:, None, None, :].expand(B, 1, N, y.shape[1])
            mask = torch.cat([causal, words], dim=-1)
        else:
            x = self.x_embedder(tokens)
            x = torch.where(known[..., None], x, self.mask_token.to(x.dtype)) + self.pos_embed.to(x.dtype)
            mask = None if y_mask is None else torch.cat([torch.ones(B, 1, 1, N, dtype=torch.bool, device=x.device), y_mask[:, None, None, :]], dim=-1)   # the official mask: all patches attend to all
        for block in self.blocks:
            x = block(x, y, condition, xpos) if mask is None else block(x, y, condition, xpos, mask)
        z = self.z_norm(x)
        if q is None:
            return z
        return self.head(q.reshape(B * N, P), tau.reshape(B * N), z.reshape(B * N, -1)).view(B, N, P)

    def _fast_head(self):
        """The head, compiled once and kept out of the module tree: it runs thousands of times per batch at sampling."""
        if not hasattr(self, "_compiled_head"):
            fast = self.head if os.environ.get("GPIC_AR_NO_COMPILE") == "1" else torch.compile(self.head)
            object.__setattr__(self, "_compiled_head", fast)
        return self._compiled_head

    def _read(self, patches):
        """What the backbone reads of a patch. With context_noise = 0 it takes no random numbers."""
        return patches + self.context_noise * torch.randn_like(patches) if self.context_noise else patches

    @torch.no_grad()
    def draw_patch(self, z):
        """One draw from the distribution of one patch given z [B, D]: the head's flow from noise, on a uniform grid, with the last step on x_hat."""
        B, head = z.shape[0], self._fast_head()
        grid = torch.linspace(0, 1, self.head_steps + 1, device=z.device)
        q = self.temperature * torch.randn(B, self.patch_dim, device=z.device)
        if os.environ.get("GPIC_AR_HEAD_FLOW") == "1":   # the whole flow in one compiled graph
            if getattr(self, "_flow_steps", None) != self.head_steps:
                flow = HeadFlow(self.head, self.head_steps, self.t_eps)
                object.__setattr__(self, "_flow", flow if os.environ.get("GPIC_AR_NO_COMPILE") == "1" else torch.compile(flow))
                object.__setattr__(self, "_flow_steps", self.head_steps)
            return self._flow(q, z).clamp(-1, 1)
        for j in range(self.head_steps):
            hat = head(q, grid[j].repeat(B), z).float()
            q = hat if j == self.head_steps - 1 else q + (grid[j + 1] - grid[j]) * (hat - q) / (1 - grid[j]).clamp_min(self.t_eps)
        return q.clamp(-1, 1)   # as real patches

    @torch.no_grad()
    def ar_sample(self, y, y_mask=None, prefix=None):
        """Images [B, C, H, W] for captions y: one patch per step, each drawn from its distribution given the patches before it. The whole sequence goes through the backbone at each step: the mask hides what is not drawn yet, and the shape stays fixed for the compiled graph.
        prefix [B, k, P] (diagnostics): the first k patches are given, and the rest are drawn. They are returned unchanged."""
        if self.order == "random":
            if self.eta and prefix is None:
                return self.sample_eta(y, y_mask, eta=self.eta)[0]
            if self.redraw_first and prefix is None:
                image, order = self._sample_random(y, y_mask, return_order=True)
                N, side = self.num_patches, image.shape[-1]
                rank = torch.zeros_like(order)
                rank.scatter_(1, order, torch.arange(N - 1, -1, -1, device=y.device).repeat(y.shape[0], 1))   # 0 = drawn first
                tokens = torch.nn.functional.unfold(image, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
                return self.unpatchify(self.redraw(tokens, rank < self.redraw_first, y, y_mask), side, side)
            return self._sample_random(y, y_mask, prefix)
        B, N = y.shape[0], self.num_patches
        tokens = torch.zeros(B, N, self.patch_dim, device=y.device)   # what comes out
        context = torch.zeros(B, N, self.patch_dim, device=y.device)  # what the backbone reads
        k = 0
        if prefix is not None:
            k = prefix.shape[1]
            tokens[:, :k] = prefix
            context[:, :k] = self._read(prefix)
        for i in range(k, N):
            z = self(context, y, y_mask=y_mask)[:, i].float()
            drawn = self.draw_patch(z)
            tokens[:, i] = drawn
            context[:, i] = self._read(drawn)
        side = int(N ** 0.5) * self.patch_size
        return self.unpatchify(tokens, side, side)

    @torch.no_grad()
    def redraw(self, tokens, pick, y, y_mask=None):
        """Draws the picked tokens again given all the others, in one step. tokens [B, N, P], pick [B, N] with the same count in each row. The training saw at most 30% known, so this asks the model about more known tokens than it trained on."""
        out = tokens.clone()
        z = self(self._read(tokens), y, y_mask=y_mask, known=~pick)
        where = pick.nonzero(as_tuple=True)
        out[where] = self.draw_patch(z[where].float())
        return out

    @torch.no_grad()
    def sample_eta(self, y, y_mask=None, eta=0.0):
        """The mask-state jump process with stochasticity eta (Campbell et al., arXiv 2402.04997, appendix F.1.1: R^DB takes a revealed token back to the mask state at rate eta per unit t, where t is the revealed share). MAR's schedule sets how many tokens stay masked after each step. At each step, each known token first goes back to the mask state with probability eta * (increase of the revealed share in this step); then, in each image, tokens chosen at random among all masked ones are drawn given the known ones until the schedule's count is met. The number of backbone calls stays ar_steps. eta = 0 is MAR's sampler in law. Returns the images and the number of head draws per image."""
        B, N = y.shape[0], self.num_patches
        tokens = torch.zeros(B, N, self.patch_dim, device=y.device)
        context = torch.zeros(B, N, self.patch_dim, device=y.device)
        unknown = torch.ones(B, N, dtype=torch.bool, device=y.device)
        draws = 0
        revealed = lambda step: 1 - math.cos(math.pi / 2 * step / self.ar_steps)
        for step in range(self.ar_steps):
            if eta:
                back = torch.rand(B, N, device=y.device) < eta * (revealed(step + 1) - revealed(step))
                unknown = unknown | back
            left = unknown.sum(1)
            if step == self.ar_steps - 1:
                count = left
            else:
                target = math.floor(N * math.cos(math.pi / 2 * (step + 1) / self.ar_steps))
                count = left - (left - 1).clamp_max(target).clamp_min(1)   # at least one token is drawn, and at least one stays masked, as in _sample_random
            if not count.any():
                continue
            keys = torch.rand(B, N, device=y.device).masked_fill(~unknown, 2.0)
            pick = keys.argsort(1).argsort(1) < count[:, None]   # the count smallest keys, all of them on masked tokens
            z = self(context, y, y_mask=y_mask, known=~unknown)
            where = pick.nonzero(as_tuple=True)
            drawn = self.draw_patch(z[where].float())
            tokens[where] = drawn
            context[where] = self._read(drawn)
            unknown = unknown & ~pick
            draws += where[0].numel()
        side = int(N ** 0.5) * self.patch_size
        return self.unpatchify(tokens, side, side), draws / B

    @torch.no_grad()
    def _sample_random(self, y, y_mask=None, prefix=None, return_order=False):
        """MAR's sampler: a new random order per image, and a cosine schedule of the unknown share from 1 to 0 over ar_steps steps. At each step the next tokens in the order are drawn given the known ones, each from its distribution. All images of a batch draw the same number of tokens per step."""
        B, N = y.shape[0], self.num_patches
        k = 0 if prefix is None else prefix.shape[1]
        tokens = torch.zeros(B, N, self.patch_dim, device=y.device)
        context = torch.zeros(B, N, self.patch_dim, device=y.device)
        unknown = torch.ones(B, N, dtype=torch.bool, device=y.device)
        keys = torch.rand(B, N, device=y.device)
        if k:
            tokens[:, :k] = prefix
            context[:, :k] = self._read(prefix)
            unknown[:, :k] = False
            keys[:, :k] += 2   # given patches come last in the order, so they never count as next
        order = keys.argsort(dim=1)
        for step in range(self.ar_steps):
            left = int(unknown[0].sum())
            if left == 0:
                break
            next_left = 0 if step == self.ar_steps - 1 else max(1, min(left - 1, math.floor(N * math.cos(math.pi / 2 * (step + 1) / self.ar_steps))))
            if next_left >= left:   # more steps than tokens: nothing to draw at this step
                continue
            after = torch.zeros_like(unknown)
            after.scatter_(1, order[:, :next_left], True)
            pick = unknown & ~after
            z = self(context, y, y_mask=y_mask, known=~unknown)
            where = pick.nonzero(as_tuple=True)
            drawn = self.draw_patch(z[where].float())
            tokens[where] = drawn
            context[where] = self._read(drawn)
            unknown = after
        side = int(N ** 0.5) * self.patch_size
        image = self.unpatchify(tokens, side, side)
        return (image, order) if return_order else image   # the token at order[:, -1] is drawn first, at order[:, -2] next
