#!/usr/bin/env python3
# The noise floor of the energy distance at 2000 samples: real against real, by dataset. usage: floor.py
import math, torch
torch.set_num_threads(1)
torch.manual_seed(7)
CELLS = torch.tensor([(i, j) for i in range(4) for j in range(4) if (i + j) % 2 == 0], dtype=torch.float32)
HARM = torch.arange(1, 9, dtype=torch.float32)
PHASE = torch.tensor([0.3, 2.1, 4.4, 1.2, 5.6, 3.3, 0.9, 2.7])
QS = torch.linalg.qr(torch.randn(8, 32, 2, generator=torch.Generator().manual_seed(123))).Q
def data(d, n):
    if d in ("manifold", "embedded"):
        ang = (torch.rand(n, 1) * 2 * math.pi) * HARM + PHASE
        xy = 0.75 * torch.stack([ang.cos(), ang.sin()], -1)
        x = 4 * (QS @ xy[..., None])[..., 0] if d == "embedded" else xy
        return x + 0.01 * torch.randn_like(x)
    D = 1 if d == "single" else 8
    idx = torch.randint(8, (n, 1)).expand(n, D) if d == "coupled" else torch.randint(8, (n, D))
    return (CELLS[idx] + torch.rand(n, D, 2)) / 2 - 1
def ed(a, b):
    a, b = a.flatten(1), b.flatten(1)
    n, m = len(a), len(b)
    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).sum() / (n * (n - 1)) - torch.cdist(b, b).sum() / (m * (m - 1))).item()
for d in ("single", "independent", "coupled", "manifold", "embedded"):
    v = torch.tensor([ed(data(d, 2000), data(d, 2000)) for _ in range(12)])
    print(f"{d:12s} real vs real, 2000 samples: mean {v.mean():.4f}, std {v.std():.4f}; 3-seed mean has std {v.std() / 3 ** 0.5:.4f}")
