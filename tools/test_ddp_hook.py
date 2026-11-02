"""C8 check: DDP gradients with allreduce_avg_hook equal the default reducer's, bit for bit.
torchrun --nproc_per_node 2 tools/test_ddp_hook.py   (PYTHONPATH=baselines/PixelGen)"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.models.transformer.JiT_T2I import JiT_T2I
from src.utils.ddp_hooks import allreduce_avg_hook

dist.init_process_group("nccl"); r = dist.get_rank(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); dev = torch.device("cuda")
kw = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=256, num_blocks=2, num_text_blocks=1, patch_size=16,
          txt_embed_dim=128, txt_max_length=16, bottleneck_dim=32)
grads = {}
for name in ("default", "avg_hook"):
    torch.manual_seed(0)
    m = JiT_T2I(**kw).to(dev)
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.02)
    d = DDP(m, device_ids=[dev.index], gradient_as_bucket_view=True, broadcast_buffers=False, bucket_cap_mb=1)
    if name == "avg_hook":
        d.register_comm_hook(None, allreduce_avg_hook)
    g = torch.Generator(device="cpu").manual_seed(100 + r)   # different data per rank
    x = torch.randn(4, 3, 64, 64, generator=g).to(dev); t = torch.rand(4, generator=g).to(dev); y = torch.randn(4, 16, 128, generator=g).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = d(x, t, y).float().pow(2).mean()
    loss.backward()
    grads[name] = [p.grad.detach().clone() for p in m.parameters() if p.requires_grad]
same = all(torch.equal(a, b) for a, b in zip(grads["default"], grads["avg_hook"]))
mx = max((a - b).abs().max().item() for a, b in zip(grads["default"], grads["avg_hook"]))
print(f"rank {r}: grads identical {same}, max |diff| {mx:.3e}, n tensors {len(grads['default'])}", flush=True)
dist.destroy_process_group()
