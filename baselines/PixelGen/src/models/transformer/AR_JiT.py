"""An image model that works as an LLM does (docs/autoregressive.md): at each step it gives the distribution of the next patch given the caption and the patches before it, and the sampler draws from it.

p(x | y) = prod_i p(patch_i | patch_1 ... patch_{i-1}, y), in raster order. The backbone is JiT_T2I's, with a causal mask: z_i sees the caption and the patches before i. There is no noise level, and the backbone's timestep embedding sees a constant.
The distribution of one patch is continuous: a small flow on the 768 values of the patch, given z_i (TokenFlowHead), with the official recipe's x-prediction and v-space L2.
"""
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


class ARJiT(JiT_T2I):
    is_ar = True

    def __init__(self, head_width=1024, head_depth=3, head_steps=32, t_eps=0.05, **kwargs):
        super().__init__(**kwargs)
        self.patch_dim = self.in_channels * self.patch_size ** 2
        self.head_steps, self.t_eps = head_steps, t_eps
        self.z_norm = self.final_layer.norm_final
        del self.final_layer   # the output is the head's; an unused parameter would stop DDP
        self.bos = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        self.head = TokenFlowHead(self.patch_dim, self.hidden_size, head_width, head_depth)
        self.register_buffer("causal", torch.tril(torch.ones(self.num_patches, self.num_patches, dtype=torch.bool)), persistent=False)

    def forward(self, tokens, y, y_mask=None, q=None, tau=None):
        """tokens [B, N, P]: the patches in raster order. The output at i sees tokens before i only, whatever tokens[:, i:] holds.
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
        x = self.x_embedder(tokens[:, :-1])
        x = torch.cat([self.bos.expand(B, 1, self.hidden_size).to(x.dtype), x], dim=1) + self.pos_embed
        causal = self.causal[None, None].expand(B, 1, N, N)   # SDPA boolean mask, True = attend
        words = torch.ones(B, 1, N, y.shape[1], dtype=torch.bool, device=x.device) if y_mask is None else y_mask[:, None, None, :].expand(B, 1, N, y.shape[1])
        mask = torch.cat([causal, words], dim=-1)
        for block in self.blocks:
            x = block(x, y, condition, xpos, mask)
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

    @torch.no_grad()
    def draw_patch(self, z):
        """One draw from the distribution of one patch given z [B, D]: the head's flow from noise, on a uniform grid, with the last step on x_hat."""
        B, head = z.shape[0], self._fast_head()
        grid = torch.linspace(0, 1, self.head_steps + 1, device=z.device)
        q = torch.randn(B, self.patch_dim, device=z.device)
        for j in range(self.head_steps):
            hat = head(q, grid[j].repeat(B), z).float()
            q = hat if j == self.head_steps - 1 else q + (grid[j + 1] - grid[j]) * (hat - q) / (1 - grid[j]).clamp_min(self.t_eps)
        return q.clamp(-1, 1)   # a drawn patch goes back in as input, and real inputs lie in [-1, 1]

    @torch.no_grad()
    def ar_sample(self, y, y_mask=None):
        """Images [B, C, H, W] for captions y: one patch per step, each drawn from its distribution given the patches before it. The whole sequence goes through the backbone at each step: the mask hides what is not drawn yet, and the shape stays fixed for the compiled graph."""
        B, N = y.shape[0], self.num_patches
        tokens = torch.zeros(B, N, self.patch_dim, device=y.device)
        for i in range(N):
            z = self(tokens, y, y_mask=y_mask)[:, i].float()
            tokens[:, i] = self.draw_patch(z)
        side = int(N ** 0.5) * self.patch_size
        return self.unpatchify(tokens, side, side)
