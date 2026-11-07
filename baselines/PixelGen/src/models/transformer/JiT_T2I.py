import math
import torch
import torch.nn as nn
import numpy as np

from functools import lru_cache
from src.models.layers.attention_op import attention
from src.models.layers.rope import apply_rotary_emb, apply_rotary_emb_real, precompute_freqs_cis_ex2d as precompute_freqs_cis_2d
from src.models.layers.time_embed import TimestepEmbedder as TimestepEmbedder
from src.models.layers.patch_embed import Embed as Embed
from src.models.layers.swiglu import SwiGLU as FeedForward
from src.models.layers.rmsnorm import RMSNorm as Norm

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Generate 2D sin-cos positional embedding.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_x = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.kv_y = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y, pos, mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv_x = self.qkv_x(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q.contiguous())
        kx = self.k_norm(kx.contiguous())
        q, kx = apply_rotary_emb_real(q, kx, pos[0], pos[1])   # perf C4: pos = (cos, sin)
        kv_y = self.kv_y(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        ky, vy = kv_y[0], kv_y[1]
        ky = self.k_norm(ky.contiguous())

        k = torch.cat([kx, ky], dim=2)
        v = torch.cat([vx, vy], dim=2)

        q = q.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        k = k.view(B, self.num_heads, -1, C // self.num_heads).contiguous()  # B, H, N, Hc
        v = v.view(B, self.num_heads, -1, C // self.num_heads).contiguous()

        x = attention(q, k, v) if mask is None else attention(q, k, v, attn_mask=mask)   # mask: [B,1,1,N+T]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4, ):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, y, c, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), y, pos, mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class TextRefineAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv_x = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        k = k.view(B, self.num_heads, -1, C // self.num_heads).contiguous()  # B, H, N, Hc
        v = v.view(B, self.num_heads, -1, C // self.num_heads).contiguous()
        x = attention(q, k, v) if mask is None else attention(q, k, v, attn_mask=mask)   # mask: [B,1,1,T]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TextRefineBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4, ):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = TextRefineAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class BottleneckEmbed(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            embed_dim: int = 768,
            bottleneck_dim: int = 256,
            bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj1 = nn.Linear(in_chans, bottleneck_dim, bias=False)
        self.proj2 = nn.Linear(bottleneck_dim, embed_dim, bias=bias)

    def forward(self, x):
        x = self.proj2(self.proj1(x))
        return x

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class EnergyHead(nn.Module):
    """Test E (docs/energy_head.md): replaces FinalLayer. The backbone's tokens h are projected once per input; then each of the m samples gets its own noise, a global vector
    through the conditioning and a vector per token, and two small residual MLP blocks per token give a whole-image sample of p(x | x_t, y)."""
    def __init__(self, hidden_size, width, out_channels, xi_dim=64, depth=2):
        super().__init__()
        self.xi_dim = xi_dim
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.inp = nn.Linear(hidden_size, width)
        self.cond = nn.Linear(hidden_size, width)
        self.xi_global = nn.Linear(xi_dim, width)
        self.xi_token = nn.Linear(xi_dim, width)
        self.blocks = nn.ModuleList(ResBlock(width) for _ in range(depth))
        self.out = FinalLayer(width, out_channels)

    def forward(self, h, c, m=1):
        B, N, _ = h.shape
        h = self.inp(self.norm(h)).repeat_interleave(m, 0)
        c = self.cond(c).repeat_interleave(m, 0)
        h = h + self.xi_token(torch.randn(B * m, N, self.xi_dim, device=h.device, dtype=h.dtype))
        c = c + self.xi_global(torch.randn(B * m, 1, self.xi_dim, device=h.device, dtype=h.dtype))
        for block in self.blocks:
            h = block(h, c)
        return self.out(h)


class CodeEncoder(nn.Module):
    """Plan A: x -> u, code_dim numbers. A small ViT on the image's patches, mean pooled. About 1% of a d640 forward with CodePrior."""
    def __init__(self, patch_dim, num_patches, code_dim, width=128, heads=4, depth=2):
        super().__init__()
        self.proj = nn.Linear(patch_dim, width)
        self.pos = nn.Parameter(torch.randn(1, num_patches, width) * 0.02)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, code_dim)

    def forward(self, tokens):
        h = self.proj(tokens) + self.pos
        for block in self.blocks:
            h = block(h)
        return self.out(self.norm(h).mean(1))


class CodePrior(nn.Module):
    """Plan A: a tiny flow on z, conditioned on the masked mean of the caption's embedding."""
    def __init__(self, code_dim, y_dim, width=1024, depth=4, y_width=256):
        super().__init__()
        self.y_norm = nn.LayerNorm(y_dim)
        self.y_proj = nn.Linear(y_dim, y_width)
        self.inp = nn.Linear(code_dim + 12 + y_width, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)) for _ in range(depth))
        self.out = nn.Linear(width, code_dim)

    def cond(self, y, y_mask=None):
        y = y.float()
        if y_mask is None:
            pooled = y.mean(1)
        else:
            m = y_mask[..., None].float()
            pooled = (y * m).sum(1) / m.sum(1).clamp_min(1)
        return self.y_proj(self.y_norm(pooled))

    def forward(self, z, tau, c):
        f = tau[:, None] * (2.0 ** torch.arange(6, device=z.device, dtype=z.dtype) * math.pi)
        h = self.inp(torch.cat([z, f.sin(), f.cos(), c.to(z.dtype)], 1))
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


class JiT_T2I(nn.Module):
    def __init__(
            self,
            input_size=256,
            in_channels=4,
            num_groups=12,
            hidden_size=1152,
            num_blocks=18,
            num_text_blocks=4,
            patch_size=2,
            txt_embed_dim=1024,
            txt_max_length=100,
            bottleneck_dim=256,
            weight_path=None,
            load_ema=False,
            xi_dim=0,
            xi_block=0,
            code_dim=0,
            code_rate=0.0,
            head_width=0,
            gm_k=0,
            gm_spread="image",
            gm_scope="pixel",
    ):
        """xi_dim > 0 makes the model an implicit sampler of p(x | x_t, y) (docs/distribution_step.md): before block xi_block the batch
        forks into m copies, and each copy gets its own noise. xi_dim = 0 is the official point model, unchanged.
        code_dim > 0 is plan A (docs/distribution_step.md 5.2): a code z of the image joins the blocks' conditioning, through a Gaussian channel
        of code_rate nats, and a tiny flow learns p(z | y). code_dim = 0 is the official point model, unchanged.
        head_width > 0 is test E (docs/energy_head.md): the backbone runs once per input and EnergyHead, in place of FinalLayer, draws m samples, each with its own noise.
        head_width = 0 is the official point model, unchanged.
        gm_k > 0 is GMFlow (docs/gmflow.md): per pixel, a mixture of gm_k Gaussians over the clean RGB value, with K x-means, K logits and one std per sample (s_x = (1 - t) s_u).
        gm_k = 0 is the official point model, unchanged.
        gm_spread (gm_k > 0 only, docs/gmflow.md section 7): "image" = one log s_u per sample, as GMFlow; "pixel" = that log s_u plus an offset per pixel from
        gm_spread_head, a zero-initialized Linear(hidden, p^2) per token on the detached input of the final linear. The offset is shared by the C channels and the
        K components of a pixel. "image" is the model before this option, unchanged.
        gm_scope (gm_k > 0 only, docs/gmflow.md section 15, owner 2026-11-06): "pixel" = K weights per pixel, the model before this option, unchanged; "image" = one
        mixture per image: the K per-pixel logits are averaged over the image before the softmax, so each image has one set of K weights (no new parameter), and
        the K means are K whole candidate images. Needs gm_spread "image"."""
        super().__init__()
        assert not (head_width and xi_dim), "the energy head and the implicit fork are two different models"
        assert not (gm_k and (head_width or xi_dim or code_dim)), "the GMFlow head is a model of its own"
        assert gm_spread in ("image", "pixel") and (gm_spread == "image" or gm_k), (gm_spread, gm_k)
        assert gm_scope in ("pixel", "image") and (gm_scope == "pixel" or (gm_k and gm_spread == "image")), (gm_scope, gm_k, gm_spread)
        self.gm_k = gm_k
        self.gm_spread = gm_spread
        self.gm_scope = gm_scope
        self.head_width = head_width
        self.xi_dim, self.xi_block = xi_dim, xi_block
        self.code_dim, self.code_rate = code_dim, code_rate
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.num_text_blocks = num_text_blocks
        self.patch_size = patch_size
        self.txt_embed_dim = txt_embed_dim
        self.txt_max_length = txt_max_length
        self.x_embedder = BottleneckEmbed(in_channels*patch_size**2, hidden_size, bottleneck_dim, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = Embed(txt_embed_dim, hidden_size, bias=True, norm_layer=Norm)
        
        # Image Positional Embedding (Fixed Sin-Cos)
        self.num_patches = (input_size//patch_size)**2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        
        self.y_pos_embedding = torch.nn.Parameter(
            torch.randn(1, txt_max_length, hidden_size)* 0.02,
            requires_grad=True
        )

        self.blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_blocks)
        ])
        
        self.text_refine_blocks = nn.ModuleList([
            TextRefineBlock(self.hidden_size, self.num_groups) for _ in range(self.num_text_blocks)
        ])
        if head_width:
            self.final_layer = EnergyHead(self.hidden_size, head_width, in_channels*patch_size**2)
        elif gm_k:   # per pixel: K means of in_channels values and K logits (GMFlow, lib/models/architecture/gmflow.py:261, 296)
            self.final_layer = FinalLayer(self.hidden_size, gm_k * (in_channels + 1) * patch_size**2)
            # log s_u per sample from the detached time and caption embeddings (GMFlow gmflow.py:179-190, 403)
            self.gm_logstd = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 1024), nn.SiLU(), nn.Linear(1024, 1))
        else:
            self.final_layer = FinalLayer(self.hidden_size, in_channels*patch_size**2)
        if xi_dim:
            self.xi_global = nn.Linear(xi_dim, hidden_size)   # one noise vector per sample, through the blocks' conditioning: the whole image
            self.xi_token = nn.Linear(xi_dim, hidden_size)    # one noise vector per token: local detail
        if code_dim:
            self.code_enc = CodeEncoder(in_channels * patch_size ** 2, self.num_patches, code_dim)
            self.code_proj = nn.Linear(code_dim, hidden_size)
            self.code_prior = CodePrior(code_dim, txt_embed_dim)

        self.initialize_weights()
        if gm_spread == "pixel":   # after initialize_weights and with the CPU RNG restored, so every other parameter and every later draw are the "image" model's;
            with torch.random.fork_rng(devices=[]):   # zero, so at init it is that model
                self.gm_spread_head = nn.Linear(hidden_size, patch_size ** 2)
            nn.init.constant_(self.gm_spread_head.weight, 0)
            nn.init.constant_(self.gm_spread_head.bias, 0)
        self.precompute_pos = dict()
        self.weight_path = weight_path
        self.load_ema = load_ema

    def fetch_pos(self, height, width, device):
        """(cos, sin) of the 2D RoPE angles, [N, Hc/2] float32 each (perf C4: real parts of freqs_cis)."""
        # perf C7: cached on the device. A CPU cache made every forward copy both tables from pageable memory, and a
        # pageable copy blocks the host until the GPU has drained its queue (35-41 ms per step, 2026-11-01 profile).
        key = (height, width, torch.device(device))
        if key not in self.precompute_pos:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width)
            self.precompute_pos[key] = (pos.real.contiguous().float().to(device), pos.imag.contiguous().float().to(device))
        return self.precompute_pos[key]

    def initialize_weights(self):
        # Basic initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize fixed sin-cos pos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))    
        nn.init.constant_(self.x_embedder.proj2.bias, 0)
    

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        for block in self.text_refine_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers
        out = self.final_layer.out.linear if self.head_width else self.final_layer.linear
        if self.gm_k:   # zero bias and a uniform weight, as GMFlow's released model (with zero weights the K means start equal and never separate). The bound is
            # Xavier's for K = 8 at every K, so the K = 1 control starts at the same output scale as K = 8 (review F5: K = 1's own Xavier gave std 0.88 against 0.38)
            b = math.sqrt(6.0 / (self.hidden_size + 8 * (self.in_channels + 1) * self.patch_size ** 2))
            nn.init.uniform_(out.weight, -b, b)
            nn.init.constant_(self.gm_logstd[-1].weight, 0)   # s_u = 1 at init (GMFlow gmflow.py:199)
            nn.init.constant_(self.gm_logstd[-1].bias, 0)
        else:
            nn.init.constant_(out.weight, 0)
        nn.init.constant_(out.bias, 0)
        if self.head_width:   # the head's blocks start as the identity, as the backbone's; the noise layers small, not zero, as xi_global below
            for block in self.final_layer.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            for layer in (self.final_layer.xi_global, self.final_layer.xi_token):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.constant_(layer.bias, 0)
        if self.xi_dim:   # small, not zero: with zero the samples start equal, and the energy score's repulsion has no direction
            for layer in (self.xi_global, self.xi_token):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.constant_(layer.bias, 0)
        if self.code_dim:
            nn.init.constant_(self.code_proj.weight, 0)   # the model starts as the point model. With std 0.02, ||z|| = sqrt(d) would put per-image noise about 3 times the timestep embedding into adaLN (review 2026-11-04)
            nn.init.constant_(self.code_proj.bias, 0)
            nn.init.constant_(self.code_prior.out.weight, 0)
            nn.init.constant_(self.code_prior.out.bias, 0)

    def code_encode(self, x):
        """Plan A: z = a u_hat + sqrt(1 - a^2) eps, u_hat = u scaled to the sphere of radius sqrt(d), a^2 = 1 - exp(-2 R / d). A Gaussian channel with power a^2 and noise 1 - a^2 per
        dimension: I(x; z) <= R nats by Shannon's bound, whatever the encoder does. So there is no KL term and no weight, and only the flow can sample z well."""
        tokens = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        u = self.code_enc(tokens).float()
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.code_dim ** 0.5
        a2 = 1 - math.exp(-2 * self.code_rate / self.code_dim)
        return a2 ** 0.5 * u + (1 - a2) ** 0.5 * torch.randn_like(u)

    def code_prior_loss(self, z, y, y_mask=None):
        """Flow matching on z given the caption. Call it with z detached: the flow must not change the code."""
        c = self.code_prior.cond(y, y_mask)
        tau = torch.rand(z.shape[0], device=z.device, dtype=z.dtype)
        eta = torch.randn_like(z)
        z_tau = tau[:, None] * z + (1 - tau[:, None]) * eta
        return ((self.code_prior(z_tau, tau, c).float() - (z - eta)) ** 2).mean()

    def code_flow(self, z0, y, y_mask=None, steps=64):
        """z ~ p(z | y) from the start z0 ~ N(0, I)."""
        c = self.code_prior.cond(y, y_mask)
        z = z0.float()
        for j in range(steps):
            z = z + self.code_prior(z, torch.full((z.shape[0],), j / steps, device=z.device, dtype=z.dtype), c).float() / steps
        return z

    def unpatchify(self, x, H, W):
        """
        x: (N, T, patch_size**2 * C)
        H, W: original height and width
        """
        x = torch.nn.functional.fold(
            x.transpose(1, 2).contiguous(), 
            output_size=(H, W), 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        return x
    
    def gm_unpatchify(self, x, H, W):
        """x: (B, T, p^2 K (C + 1)) -> means (B, K, C, H, W) and raw logits (B, K, 1, H, W). fold reads each token as [channel, row, column], the order unfold writes."""
        B, K, C = x.shape[0], self.gm_k, self.out_channels
        x = self.unpatchify(x, H, W)
        means, logits = x.split([K * C, K], dim=1)
        return means.reshape(B, K, C, H, W), logits.reshape(B, K, 1, H, W)

    def forward(self, x, t, y, return_layer=None, y_mask=None, m=1, z=None, gm_mode="mean"):
        """y_mask (ladder option B): [B, T] bool, True = a real caption token, T <= txt_max_length; None = the
        official unmasked path (y padded to txt_max_length). m: samples per input of the implicit sampler; the
        output is [B * m, C, H, W], each input's samples next to each other. z: plan A's code, [B, code_dim].
        gm_mode (gm_k > 0 only): "mean" returns the mixture mean sum_k A_k mu_k as the x prediction, so the point model's samplers run on it unchanged;
        "full" returns dict(means (B, K, C, H, W), logweights (B, K, 1, H, W), log-softmax over K, fp32, logstd (B, 1, 1, 1, 1) = log s_u, fp32; (B, 1, 1, H, W) with gm_spread "pixel").
        With gm_scope "image", logweights is (B, K, 1, 1, 1): one set of weights per image."""
        B, _, H, W = x.shape
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        xpos = self.fetch_pos(H // self.patch_size, W // self.patch_size, x.device)
        ypos = self.y_pos_embedding if y_mask is None else self.y_pos_embedding[:, :y.shape[1]]
        text_mask = full_mask = None
        if y_mask is not None:   # SDPA boolean masks, True = attend
            text_mask = y_mask[:, None, None, :]
            full_mask = torch.cat([torch.ones(B, 1, 1, x.shape[1], dtype=torch.bool, device=x.device), text_mask], dim=-1)
        t = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y_dtype = y.dtype   # the input's dtype, as before this option existed
        y = self.y_embedder(y).view(B, -1, self.hidden_size)
        if self.gm_k and gm_mode == "full":   # the std's input: time plus the mean caption embedding over the real tokens, detached
            pooled = y.mean(1) if y_mask is None else (y * y_mask[..., None].to(y.dtype)).sum(1) / y_mask.sum(1, keepdim=True).clamp_min(1).to(y.dtype)
            gm_cond = (t.view(B, self.hidden_size) + pooled).detach()
        y = y + ypos.to(y_dtype)
        condition = nn.functional.silu(t if z is None else t + self.code_proj(z.to(t.dtype)).view(B, 1, self.hidden_size))
        for i, block in enumerate(self.text_refine_blocks):
            y = block(y, condition) if text_mask is None else block(y, condition, text_mask)

        x = self.x_embedder(x)
        x += self.pos_embed
        for i, block in enumerate(self.blocks):
            if self.xi_dim and i == self.xi_block:   # the blocks before this one run once per input
                x, y, condition = x.repeat_interleave(m, 0), y.repeat_interleave(m, 0), condition.repeat_interleave(m, 0)
                full_mask = None if full_mask is None else full_mask.repeat_interleave(m, 0)
                x = x + self.xi_token(torch.randn(x.shape[0], x.shape[1], self.xi_dim, device=x.device, dtype=x.dtype))
                condition = condition + self.xi_global(torch.randn(x.shape[0], 1, self.xi_dim, device=x.device, dtype=x.dtype))
            if return_layer is not None and i==return_layer:
                feat = x
            x = block(x, y, condition, xpos) if full_mask is None else block(x, y, condition, xpos, full_mask)
            
        spread = None
        if self.gm_k and self.gm_spread == "pixel" and gm_mode == "full":   # FinalLayer has no adaLN: its linear's input is norm_final(x)
            h = self.final_layer.norm_final(x)
            x = self.final_layer.linear(h)
            spread = self.unpatchify(self.gm_spread_head(h.detach()), H, W).float().unsqueeze(1)   # (B, 1, 1, H, W): shared by the K components and the C channels
        else:
            x = self.final_layer(x, condition, m) if self.head_width else self.final_layer(x)
        if self.gm_k:
            assert return_layer is None and gm_mode in ("mean", "full"), gm_mode
            means, logits = self.gm_unpatchify(x, H, W)
            if self.gm_scope == "image":   # one mixture per image: logweights (B, K, 1, 1, 1)
                logits = logits.float().mean((-2, -1), keepdim=True)
            logweights = logits.float().log_softmax(dim=1)
            if gm_mode == "mean":
                return (logweights.exp() * means.float()).sum(1)
            logstd = self.gm_logstd(gm_cond).float().view(B, 1, 1, 1, 1)
            return dict(means=means, logweights=logweights, logstd=logstd if spread is None else logstd + spread)
        output = self.unpatchify(x, H, W)
        if return_layer is not None:
            return output, feat
        else:
            return output