"""Tests of src/optim/muon.py (MuonAdamW) and the tagging, on small JiT_T2I models with synthetic gradients.

single GPU : PYTHONPATH=baselines/PixelGen python tools/test_muon.py
two ranks  : PYTHONPATH=baselines/PixelGen torchrun --nproc_per_node 2 tools/test_muon.py
Checks: msign against the exact polar factor; MuonAdamW against a per-matrix reference (Keller Jordan's update with
the Moonlight scale) and torch's fused AdamW; update RMS ~ 0.2 lr; split parts; state_dict round trip; with two ranks,
identical weights on both ranks and equal to the single-process result; the tagged set on the full-size model.
"""
import copy
import math
import os

import torch
import torch.distributed as dist

from src.models.transformer.JiT_T2I import JiT_T2I
from src.optim.muon import MuonAdamW, msign, tag_muon_params

SMALL = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=256, num_blocks=2, num_text_blocks=1, patch_size=16,
             txt_embed_dim=128, txt_max_length=16, bottleneck_dim=32)
FULL = dict(input_size=256, in_channels=3, num_groups=24, hidden_size=1536, num_blocks=16, num_text_blocks=4,
            patch_size=16, txt_embed_dim=2048, txt_max_length=300, bottleneck_dim=256)


def model(dev, seed=0):
    torch.manual_seed(seed)
    m = JiT_T2I(**SMALL).to(dev)
    for p in m.parameters():   # break the zero init so every matrix has a non-trivial gradient path
        p.data.add_(torch.randn_like(p) * 0.02)
    tag_muon_params(m)
    return m


def set_grads(m, step):
    g = torch.Generator(device="cpu").manual_seed(1000 + step)
    for p in m.parameters():
        p.grad = torch.randn(p.shape, generator=g).to(p.device) * 1e-2


def reference_step(m, state, lr, beta=0.95, adam=None, split=False, dtype=torch.float32):
    """Keller Jordan's muon_update per matrix (or per part), Moonlight scale, applied matrix by matrix."""
    for p in m.parameters():
        if not hasattr(p, "_muon_parts"):
            continue
        buf = state.setdefault(p, torch.zeros_like(p))
        buf.lerp_(p.grad, 1 - beta)
        u = p.grad.lerp(buf, beta)
        parts = p._muon_parts if split else 1
        rows = p.shape[0] // parts
        for k in range(parts):
            o = msign(u[k * rows:(k + 1) * rows][None], dtype=dtype)[0]
            p.data[k * rows:(k + 1) * rows].add_(o, alpha=-lr * 0.2 * math.sqrt(max(rows, p.shape[1])))
    adam.step()


def max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a.parameters(), b.parameters()))


def main():
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank() if ddp else 0
    dev = torch.device("cuda")
    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    ok = True

    # 1. msign against the polar factor U V^T from an SVD
    torch.manual_seed(0)
    for shape in [(256, 256), (768, 256), (256, 1024)]:
        G = torch.randn(shape, device=dev)
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        O = msign(G[None])[0].float()
        sv = torch.linalg.svdvals(O)
        cos = (O * (U @ Vh)).sum() / (O.norm() * (U @ Vh).norm())
        good = cos > 0.95 and sv.max() < 1.25 and (shape[0] == shape[1] or sv.min() > 0.6)   # a square Gaussian has near-zero singular values that 5 steps do not lift
        ok &= bool(good)
        say(f"msign {shape}: cos to UV^T {cos:.4f}, singular values [{sv.min():.3f}, {sv.max():.3f}]  {'ok' if good else 'FAIL'}")

    for split in (False, True):
        lr = 2e-3
        a, b = model(dev), model(dev)
        assert max_diff(a, b) == 0
        opt = MuonAdamW([{"params": list(a.parameters())}], lr=lr, muon_lr=lr, split=split, ns_dtype="float32")
        adam_params = [p for p in b.parameters() if not hasattr(p, "_muon_parts")]
        ref_adam = torch.optim.AdamW(adam_params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
        st = {}
        w0 = {n: p.detach().clone() for n, p in a.named_parameters()}
        for s in range(3):
            set_grads(a, s); set_grads(b, s)
            opt.step(); reference_step(b, st, lr, adam=ref_adam, split=split)
            if s == 0:   # update RMS of a Muon matrix (part) from zero momentum, in units of lr
                rms = []
                for n, p in a.named_parameters():
                    if hasattr(p, "_muon_parts"):
                        parts = p._muon_parts if split else 1
                        rows = p.shape[0] // parts
                        for k in range(parts):
                            d = (p.detach() - w0[n])[k * rows:(k + 1) * rows]
                            rms.append(d.pow(2).mean().sqrt().item() / lr)
                good = all(0.12 < r < 0.25 for r in rms)
                ok &= good
                say(f"split={split}: update RMS / lr over {len(rms)} parts: [{min(rms):.3f}, {max(rms):.3f}]  {'ok' if good else 'FAIL'}")
        dm = max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()) if hasattr(p, "_muon_parts"))
        da = max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()) if not hasattr(p, "_muon_parts"))
        good = dm < 1e-6 and da == 0   # fp32 Newton-Schulz: only fp32 rounding between batched and single GEMMs
        ok &= good
        say(f"split={split}: vs per-matrix reference after 3 steps: Muon max |diff| {dm:.2e}, AdamW max |diff| {da:.2e}  {'ok' if good else 'FAIL'}")

        # state_dict round trip: continue in a fresh optimizer from the saved state
        c = copy.deepcopy(a)
        tag_muon_params(c)
        opt2 = MuonAdamW([{"params": list(c.parameters())}], lr=lr, muon_lr=lr, split=split, ns_dtype="float32")
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()))   # a checkpoint holds copies, not the live buffers
        set_grads(a, 9); set_grads(c, 9)
        opt.step(); opt2.step()
        d = max_diff(a, c)
        ok &= d == 0
        say(f"split={split}: state_dict round trip max |diff| {d:.2e}  {'ok' if d == 0 else 'FAIL'}")

        if ddp:   # identical weights on every rank, and close to a single-process run of the same steps
            ws = [None] * dist.get_world_size()
            dist.all_gather_object(ws, [p.detach().cpu() for p in a.parameters()])
            same = all(all(torch.equal(x, y) for x, y in zip(ws[0], w)) for w in ws[1:])
            ok &= same
            say(f"split={split}: weights identical on all {len(ws)} ranks: {same}")
            if rank == 0:
                # a single-process optimizer on rank 0 only (no all_gather): build the plan for world 1
                e = model(dev)
                opt1 = MuonAdamW([{"params": list(e.parameters())}], lr=lr, muon_lr=lr, split=split, ns_dtype="float32")
                opt1._plan = None
                saved = (dist.get_world_size, dist.get_rank)
                dist.get_world_size, dist.get_rank = (lambda *x: 1), (lambda *x: 0)
                try:
                    for s in list(range(3)) + [9]:
                        set_grads(e, s); opt1.step()
                finally:
                    dist.get_world_size, dist.get_rank = saved
                d1 = max_diff(a, e)
                good = d1 < 1e-6
                ok &= good
                say(f"split={split}: 2 ranks vs 1 process max |diff| {d1:.2e}  {'ok' if good else 'FAIL'}")
            dist.barrier()

    # the tagged set on the full-size model (meta device: names and shapes only)
    with torch.device("meta"):
        full = JiT_T2I(**FULL)
    n, numel = tag_muon_params(full)
    parts = sorted({(name.split(".", 2)[-1], p._muon_parts) for name, p in full.named_parameters() if hasattr(p, "_muon_parts")})
    total = sum(p.numel() for p in full.parameters() if p.requires_grad)
    good = n == 16 * 6 + 4 * 5
    ok &= good
    say(f"full model: {n} Muon matrices, {numel / 1e6:.1f}M of {total / 1e6:.1f}M trainable params  {'ok' if good else 'FAIL'}")
    say("  per-block matrices and parts:", parts)
    say("ALL OK" if ok else "SOME CHECKS FAILED")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
