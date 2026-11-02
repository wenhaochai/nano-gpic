import torch
from src.optim.muon import msign
torch.manual_seed(0)
dev = torch.device("cuda")
for shape in [(256, 256), (768, 256), (256, 1024), (512, 256)]:
    G = torch.randn(4, *shape, device=dev)
    a = msign(G); b = msign(G)
    singles = torch.stack([msign(G[i:i + 1])[0] for i in range(4)])
    print(shape, "repeat max|diff|", (a.float() - b.float()).abs().max().item(),
          " batched vs single", (a.float() - singles.float()).abs().max().item(), " max|O|", a.float().abs().max().item())
