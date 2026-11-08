#!/usr/bin/env python3
"""genmodel 2026-11-07 23:xx: the d1536 ladder checkpoints score a held-out loss of ~0.25 (eager forward) against ~0.067 at the
other widths, while their training loss is normal. Does a training-path kernel compute a different function at d1536?
Run once per environment (the switches are read at import): the same fixed held-out batch (B val images, given levels, the
held-out tool's noise) through a ladder checkpoint's raw and EMA weights, and a random-init model's forward on fixed
inputs; prints the losses and saves the outputs so two runs can be compared (compare mode).
Code: the ladder worktree given (its src, with whatever GPIC_MFU_* the environment sets).
usage: python kernel_vs_eager.py run <code worktree> <out.pt> <ckpt> [<ckpt> ...]
       python kernel_vs_eager.py compare <a.pt> <b.pt>"""
import os
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if sys.argv[1] == "compare":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    print(f"A {a['env']}\nB {b['env']}")
    for k in a["loss"]:
        if k in b["loss"]:
            print(f"  {k}: loss A {a['loss'][k]:.5f}  B {b['loss'][k]:.5f}")
    for k in a["out"]:
        if k in b["out"]:
            d = (a["out"][k].float() - b["out"][k].float())
            print(f"  {k}: output max |A - B| {d.abs().max():.3e}, rel rms {d.norm() / a['out'][k].float().norm():.3e}")
    sys.exit(0)

code, dst, ckpts = sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/gpic-wt/heldout_loss_run/tools")
import heldout_loss as H  # noqa: E402
sys.path[:] = [p for p in sys.path if not p.startswith(H.REPO)]   # the model code from <code>, not the tool's copy
sys.path.insert(0, os.path.join(code, "baselines", "PixelGen"))
sys.path.insert(1, "/scratch/gpfs/GROUP/USER/project/gpic-wt/heldout_loss_run/tools")
dev = "cuda"
B, LV = 8, [1, 5, 10, 14]
env = {k: v for k, v in os.environ.items() if k.startswith(("GPIC_MFU", "KVE_"))}
res = {"env": env, "loss": {}, "out": {}}
data = H.load_data(H.data_path(3072, 3072), "cpu")
split = data["val"]
idx = split["order"][::len(split["order"]) // B][:B]
x = H.to_input(split["img"][torch.tensor(idx)]).to(dev)
y, mask = H.text_batch(split, idx, dev)
noise = torch.stack([H.image_noise(H.SEED, j, 16, tuple(x.shape[1:])) for j in idx]).to(dev)
for ck in ckpts:
    kw, loss_cfg = H.run_config(os.path.dirname(ck))
    tr = H.make_trainer(loss_cfg)
    ts = H.levels(loss_cfg["P_mean"], loss_cfg["P_std"], loss_cfg["timeshift"])
    sd = torch.load(ck, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    tag = os.path.basename(os.path.dirname(ck))[len("exp_ladder_"):] + "_" + os.path.basename(ck)[8:14]
    for wname, prefix in H.WEIGHTS:
        net = H.build_denoiser(kw, sd, prefix, dev)
        if os.environ.get("KVE_COMPILE") == "1":   # as training under FSDP2: every text and image block compiled on its own
            for blk in list(net.text_refine_blocks) + list(net.blocks):
                blk.compile()
        with torch.no_grad():
            ls = [H.level_loss(net, tr, x, y, mask, ts[k].to(dev), noise[:, k], True).mean().item() for k in LV]
            t = ts[LV[1]].to(dev).expand(B)
            xt = (1 - t.view(-1, 1, 1, 1)) * noise[:, LV[1]] + t.view(-1, 1, 1, 1) * x   # any fixed input
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res["out"][f"{tag}_{wname}"] = net.forward(xt, t, y, y_mask=mask).float().cpu()
        res["loss"][f"{tag}_{wname}"] = sum(ls) / len(ls)
        print(f"{tag} {wname}: loss {res['loss'][f'{tag}_{wname}']:.5f} (levels {[round(v, 4) for v in ls]})", flush=True)
        del net
    if "d1536" in tag:   # a random-init model of the same shape: the same weights in every run (seeded CPU init)
        from src.models.transformer.JiT_T2I import JiT_T2I
        torch.manual_seed(0)
        net = JiT_T2I(**kw).float()
        for n, p in net.named_parameters():   # the zero-init gates and final layer would hide the blocks
            if p.ndim >= 1 and torch.count_nonzero(p) == 0:
                torch.nn.init.normal_(p, std=0.02)
        net = net.to(dev).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t = ts[LV[1]].to(dev).expand(B)
            res["out"]["d1536_random"] = net.forward(noise[:, LV[1]], t, y, y_mask=mask).float().cpu()
        del net
    del sd
torch.save(res, dst)
print("KERNEL_VS_EAGER DONE", dst)
