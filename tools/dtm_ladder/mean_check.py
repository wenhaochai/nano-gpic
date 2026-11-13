"""Check of the DTM head's mean, the quantity the held-out loss scores (docs/dtm_ladder.md): the model's default output (the head's x prediction at head
time 0, averaged over dtm_mean_draws fixed head noises) against an assumption-free estimate from the head's own draws.

For n held-out images at 4 of the tool's 16 levels (the same images, levels and noise as tools/heldout_loss.py), with v-space errors e = (x_hat - x) / (1 - t):
  mse_mean  = mean over values of e(mean)^2                                  (what the held-out tool records)
  mse_ustat = mean over pairs r != s of <e_r, e_s> / values, R head draws    (unbiased for the squared error of the draws' true mean)
  spread    = mean over values of the draws' variance                        (how wide the head's distribution is)
If the head's x prediction at time 0 is its mean, mse_mean ~ mse_ustat (up to the R-draw noise). usage (GPU):
  python tools/dtm_ladder/mean_check.py <ckpt> <data file> [n images, default 256] [R draws, default 4]
Prints one line per level and the totals; exit 0 always (a report, not a gate)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import heldout_loss as H   # noqa: E402

ckpt, dpath = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 256
R = int(sys.argv[4]) if len(sys.argv) > 4 else 4
dev = "cuda" if torch.cuda.is_available() else "cpu"
kw, loss = H.run_config(os.path.dirname(ckpt))
assert kw.get("dtm_width"), "not a DTM run"
sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
net = H.build_denoiser(kw, sd, "denoiser.", dev)
data = H.load_data(dpath, dev)
split = data["val"]
ts = H.levels(loss["P_mean"], loss["P_std"], loss["timeshift"])
idx = list(range(n))
C, Hh, Ww = 3, 256, 256
res = {}
torch.manual_seed(0)
for k in (2, 6, 10, 14):
    t = ts[k]
    sm = su = sp = 0.0
    for b0 in range(0, n, 32):
        ib = idx[b0:b0 + 32]
        it = torch.tensor(ib, device=dev)
        x = H.to_input(split["img"][it]).to(dev)
        y, mask = H.text_batch(split, ib, dev)
        noise = torch.stack([H.image_noise(H.SEED, j, len(ts), (C, Hh, Ww))[k] for j in ib]).to(dev)
        x_t = t * x + (1 - t) * noise
        s = float(max(1 - float(t), loss["t_eps"]))
        tt = t.expand(len(ib)).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            mean = net(x_t, tt, y, y_mask=mask).float()
            draws = [net(x_t, tt, y, y_mask=mask, dtm_mode="draw").float() for _ in range(R)]
        e_mean = (mean - x) / s
        es = torch.stack([(d - x) / s for d in draws])                      # [R, B, C, H, W]
        tot = es.sum(0)
        pair = (tot * tot - (es * es).sum(0)) / (R * (R - 1))               # sum over r != s of e_r e_s, per value
        sm += float((e_mean ** 2).mean(dim=(1, 2, 3)).sum())
        su += float(pair.mean(dim=(1, 2, 3)).sum())
        sp += float(torch.stack(draws).var(0, unbiased=True).mean(dim=(1, 2, 3)).sum() / s ** 2)
    res[float(t)] = dict(mse_mean=sm / n, mse_ustat=su / n, spread=sp / n)
    r = res[float(t)]
    print(f"t {float(t):.3f}: mse_mean {r['mse_mean']:.5f}, mse_ustat {r['mse_ustat']:.5f} (ratio {r['mse_mean'] / r['mse_ustat']:.4f}), "
          f"draws' variance {r['spread']:.5f}", flush=True)
m = sum(r["mse_mean"] for r in res.values()) / len(res)
u = sum(r["mse_ustat"] for r in res.values()) / len(res)
print(f"mean over the 4 levels: mse_mean {m:.5f}, mse_ustat {u:.5f}, ratio {m / u:.4f} ({n} images, {R} draws)")
print(json.dumps(dict(ckpt=ckpt, n=n, R=R, levels=res)))
