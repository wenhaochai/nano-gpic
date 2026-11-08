#!/usr/bin/env python3
"""The d1536 ladder checkpoints hold final_layer.linear's 8 row blocks (its FSDP2 Shard(0) pieces) in another order than
the training forward used (gpic-28 2026-11-08: un-permuting them restores the held-out loss). Which side permutes? Under
torchrun on 8 GPUs, with the training's FSDP2 setup (src/utils/fsdp_setup.shard, one block per unit, bf16 compute): load a
checkpoint's raw weights, run one forward, and compare, parameter by parameter, block by block (8 row blocks), (a) the
unsharded weight the module sees inside the forward, (b) DTensor.full_tensor(), (c) the checkpoint. Prints the block
permutation for every parameter where (a) or (b) differs from (c).
usage: torchrun --nproc_per_node 8 fsdp_perm_test.py <code worktree> <ms-step ckpt> [compile]"""
import os
import sys

import torch
import torch.distributed as dist

code, ck = sys.argv[1], sys.argv[2]
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/gpic-wt/heldout_loss_run/tools")
import heldout_loss as H  # noqa: E402
sys.path[:] = [p for p in sys.path if not p.startswith(H.REPO)]
sys.path.insert(0, os.path.join(code, "baselines", "PixelGen"))
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.tensor import DTensor  # noqa: E402
from src.models.transformer.JiT_T2I import JiT_T2I  # noqa: E402
from src.utils.fsdp_setup import shard  # noqa: E402

kw, _ = H.run_config(os.path.dirname(ck))
sd = torch.load(ck, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
sub = {k[len("denoiser."):]: v for k, v in sd.items() if k.startswith("denoiser.")}
with torch.device("meta"):
    net = JiT_T2I(**kw)
net.load_state_dict(sub, strict=True, assign=True)
net = net.cuda().float()
mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("data_parallel",))
shard(net, mesh, compute_bf16=True, compile_blocks=len(sys.argv) > 3, names={m: n for n, m in net.named_modules()})
watch = ["final_layer.linear", "x_embedder.proj1", "x_embedder.proj2", "t_embedder.mlp.0", "t_embedder.mlp.2",
         "y_embedder.proj", "blocks.0.attn.qkv_x", "blocks.15.mlp.w3", "text_refine_blocks.0.attn.qkv"]
mods = dict(net.named_modules())
seen = {}
for name in watch:
    def hook(m, inp, name=name):
        for pn in ("weight", "bias"):
            p = getattr(m, pn, None)
            if p is not None:
                seen[f"{name}.{pn}"] = (p.full_tensor() if isinstance(p, DTensor) else p).detach().float().cpu()
    mods[name].register_forward_pre_hook(hook)
B, T = 2, 32
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(B, 3, 256, 256, device="cuda", generator=g)
t = torch.rand(B, device="cuda", generator=g)
y = torch.randn(B, T, kw.get("txt_embed_dim", 2048), device="cuda", generator=g)
mask = torch.ones(B, T, dtype=torch.bool, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    net(x, t, y, y_mask=mask)
torch.cuda.synchronize()


def block_perm(a, ref):
    """a, ref [R, ...]: for each of 8 row blocks of a, the block of ref it equals (bf16-rounded), else -1."""
    R = ref.shape[0]
    if R % 8:
        return None
    ab, rb = a.reshape(8, R // 8, -1), ref.to(torch.bfloat16).float().reshape(8, R // 8, -1)
    out = []
    for k in range(8):
        hit = [j for j in range(8) if torch.equal(ab[k].to(torch.bfloat16).float(), rb[j])]
        out.append(hit[0] if hit else -1)
    return out


if rank == 0:
    print(f"{os.path.basename(os.path.dirname(ck))} {os.path.basename(ck)}, world {world}, compile {len(sys.argv) > 3}")
    for k in sorted(seen):
        ref = sub[k].float()
        inside = seen[k]
        full = None
        mod, pn = k.rsplit(".", 1)
        p = getattr(mods[mod], pn)
        if isinstance(p, DTensor):
            full = p.full_tensor().detach().float().cpu()
        bp_in = block_perm(inside, ref)
        bp_full = block_perm(full, ref) if full is not None else "(not a DTensor after forward)"
        same_in = torch.equal(inside.to(torch.bfloat16), ref.to(torch.bfloat16))
        print(f"  {k:36s} {str(tuple(ref.shape)):16s} forward == ckpt (bf16): {same_in}; forward blocks <- ckpt blocks {bp_in}; full_tensor blocks {bp_full}")
dist.barrier()
dist.destroy_process_group()
