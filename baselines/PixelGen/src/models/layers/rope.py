from typing import Tuple
import torch


def precompute_freqs_cis_2d(dim: int, height: int, width:int, theta: float = 10000.0, scale=16.0):
    # assert  H * H == end
    # flat_patch_pos = torch.linspace(-1, 1, end) # N = end
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(height*width, -1)
    return freqs_cis

def precompute_freqs_cis_ex2d(dim: int, height: int, width:int, theta: float = 10000.0, scale=1.0):
    if isinstance(scale, float):
        scale = (scale, scale)
    x_pos = torch.linspace(0, height*scale[0], width)
    y_pos = torch.linspace(0, width*scale[1], height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(height*width, -1)
    return freqs_cis


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, None, :, :]
    # xq : B N H Hc
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_rotary_emb_real(
        xq: torch.Tensor,
        xk: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """apply_rotary_emb in real arithmetic (perf C4): (a + ib)(c + is) = (ac - bs) + i(as + bc) on the interleaved
    pairs, in fp32 as the complex version, so Inductor fuses it instead of falling back to complex aten kernels.
    cos, sin: [N, Hc/2] float32 (the real and imaginary parts of freqs_cis)."""
    cos, sin = cos[None, None], sin[None, None]

    def rot(x):
        xf = x.float().reshape(*x.shape[:-1], -1, 2)
        x0, x1 = xf[..., 0], xf[..., 1]
        return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(3).type_as(x)

    return rot(xq), rot(xk)


def apply_rotary_emb_crossattention(
        xq: torch.Tensor,
        xk: torch.Tensor,
        yk: torch.Tensor,
        freqs_cis1: torch.Tensor,
        freqs_cis2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    freqs_cis1 = freqs_cis1[None, None, :, :]
    freqs_cis2 = freqs_cis2[None, None, :, :]
    # xq : B N H Hc
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    yk_ = torch.view_as_complex(yk.float().reshape(*yk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis1).flatten(3) # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis1).flatten(3)
    yk_out = torch.view_as_real(yk_ * freqs_cis2).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk), yk_out.type_as(yk)