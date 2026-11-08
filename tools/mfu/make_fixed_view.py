#!/usr/bin/env python3
"""Fixed view of FSDP2 ladder checkpoints whose final_layer.linear row blocks are stored out of order (gpic-28 2026-11-08:
the d1536 runs hold its 8 row blocks of 96 rows, the FSDP2 Shard(0) pieces, in another order than the training forward
used; the permutation changes between jobs). For each checkpoint, separately for the raw (denoiser.*) and EMA
(ema_denoiser.*) weights: the low-noise predictions of 32 held-out val images give, for every output block k of the 768
patch features, the target block pi[k] it reproduces (mean matched correlation; every block must match one block, above
0.9, and pi must be a permutation); then final_layer.linear.weight and .bias are re-ordered (new block j = old block k with
pi[k] = j), and the held-out loss of 8 images x 4 levels must drop. Writes runs/gpic-speed/fixed_view/<root>/exp_ladder_<size>/
ms-step=<step>.ckpt (state_dict and the keys the evaluation reads, as tmp/ladder_prof/make_raw_view.py) and copies the
run's config files. A checkpoint whose pi is the identity for both weights is copied as it is (state_dict only).
usage: make_fixed_view.py <root under runs/gpic-speed> <size> <step> [<step> ...]"""
import glob
import os
import shutil
import sys

import torch

sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/gpic-wt/heldout_loss_run/tools")
import heldout_loss as H  # noqa: E402
CODE = "/scratch/gpfs/GROUP/USER/project/gpic-wt/muon_ladder_run2"   # d1536's C0 code (every forward path agrees)
sys.path[:] = [p for p in sys.path if not p.startswith(H.REPO)]
sys.path.insert(0, os.path.join(CODE, "baselines", "PixelGen"))
torch.backends.cuda.matmul.allow_tf32 = False
R = "/scratch/gpfs/GROUP/USER/runs/gpic-speed"
root, size, steps = sys.argv[1].strip("/"), sys.argv[2], [int(s) for s in sys.argv[3:]]
assert steps, __doc__
dev = "cuda"
NAMES = ("final_layer.linear.weight", "final_layer.linear.bias")
data = H.load_data(H.data_path(3072, 3072), "cpu")
split = data["val"]
idx32 = split["order"][::96][:32]
x32 = H.to_input(split["img"][torch.tensor(idx32)]).to(dev)
y32, m32 = H.text_batch(split, idx32, dev)
idx8 = split["order"][::len(split["order"]) // 8][:8]
x8 = H.to_input(split["img"][torch.tensor(idx8)]).to(dev)
y8, m8 = H.text_batch(split, idx8, dev)
n8 = torch.stack([H.image_noise(H.SEED, j, 16, tuple(x8.shape[1:])) for j in idx8]).to(dev)


def patches(z):
    return torch.nn.functional.unfold(z, 16, stride=16).transpose(1, 2).reshape(-1, 768)


def reorder(sd, prefix, pi):
    out = dict(sd)
    for n in NAMES:
        w = sd[prefix + n]
        blocks = list(w.reshape(8, w.shape[0] // 8, *w.shape[1:]))
        out[prefix + n] = torch.cat([blocks[pi.index(j)] for j in range(8)], dim=0).contiguous()
    return out


for step in steps:
    src = f"{R}/{root}/exp_ladder_{size}/ms-step={step:06d}.ckpt"
    out_dir = f"{R}/fixed_view/{root}/exp_ladder_{size}"
    if os.environ.get("FIXED_VIEW_CHECK") != "1":
        os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/ms-step={step:06d}.ckpt"
    if os.path.exists(out) and os.environ.get("FIXED_VIEW_CHECK") != "1":
        print("exists", out)
        continue
    kw, lc = H.run_config(os.path.dirname(src))
    ts = H.levels(lc["P_mean"], lc["P_std"], lc["timeshift"])
    tr = H.make_trainer(lc)
    ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["state_dict"]
    assert all(sd[p + n].shape[0] % 8 == 0 for p in ("denoiser.", "ema_denoiser.") for n in NAMES)
    new, perms = dict(sd), {}
    for wname, prefix in H.WEIGHTS:
        net = H.build_denoiser(kw, sd, prefix, dev)
        lv = len(ts) - 1
        noise = torch.stack([H.image_noise(H.SEED, j, 16, tuple(x32.shape[1:]))[lv] for j in idx32]).to(dev)
        t = ts[lv].to(dev).expand(len(idx32))
        xt = t.view(-1, 1, 1, 1) * x32 + (1 - t.view(-1, 1, 1, 1)) * noise
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net.forward(xt, t, y32, y_mask=m32).float()
        a, b = patches(pred), patches(x32)
        a, b = (a - a.mean(0)) / a.std(0), (b - b.mean(0)) / b.std(0)
        C = (a.t() @ b) / a.shape[0]
        blk = C.reshape(8, 96, 8, 96).diagonal(dim1=1, dim2=3).mean(-1)
        pi = blk.argmax(dim=1).tolist()
        corr = [round(float(blk[k, pi[k]]), 3) for k in range(8)]
        assert sorted(pi) == list(range(8)) and min(corr) > 0.9, (wname, pi, corr)
        perms[wname] = pi
        del net

        def loss(s):
            n = H.build_denoiser(kw, s, prefix, dev)
            with torch.no_grad():
                v = sum(H.level_loss(n, tr, x8, y8, m8, ts[k].to(dev), n8[:, k], True).mean().item() for k in (1, 5, 10, 14)) / 4
            del n
            return v
        if pi != list(range(8)):
            fixed = reorder(sd, prefix, pi)
            l0, l1 = loss(sd), loss(fixed)
            assert l1 < l0, (wname, pi, l0, l1)
            for n in NAMES:
                new[prefix + n] = fixed[prefix + n]
            print(f"{size} {step} {wname}: pi {pi} (block corr {corr}); held-out loss {l0:.4f} -> {l1:.4f}", flush=True)
        else:
            print(f"{size} {step} {wname}: pi is the identity (block corr {corr})", flush=True)
    if os.environ.get("FIXED_VIEW_CHECK") == "1":   # only estimate and print pi
        continue
    keep = {k: v for k, v in ck.items() if k not in ("optimizer_states", "lr_schedulers", "callbacks", "loops")}
    keep["state_dict"] = new
    keep["fixed_view_of"] = src
    keep["final_layer_block_perm"] = perms
    tmp = out + ".tmp"
    torch.save(keep, tmp)
    os.replace(tmp, out)
    for c in glob.glob(f"{R}/{root}/exp_ladder_{size}/config-*.yaml"):
        dst = os.path.join(out_dir, os.path.basename(c))
        if not os.path.exists(dst):
            shutil.copy(c, dst)
    print("wrote", out, f"{os.path.getsize(out) / 2**30:.2f} GiB", flush=True)
print("MAKE_FIXED_VIEW DONE")
