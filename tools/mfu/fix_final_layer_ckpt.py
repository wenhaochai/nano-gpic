#!/usr/bin/env python3
"""Re-order a full-state ladder checkpoint (last.ckpt) whose final_layer.linear.weight was saved with its 8 row blocks
permuted by NCCL 2.26.2's NVLS all-gather (gpic-28 2026-11-08; NVIDIA/nccl #1906): the weight in denoiser.* and
ema_denoiser.*, and its two Adam moments in optimizer_states (the same size, gathered the same way), so that a resume
continues the trained model exactly. pi (from tools/mfu/make_fixed_view.py on the matching ms-step file): output block k
of the saved weight holds the trained block pi[k]; new block j = old block k with pi[k] = j. The bias is not touched (it
was saved in order). Writes <out> (never the input).
usage: fix_final_layer_ckpt.py <in last.ckpt> <out> <pi, e.g. 0,3,1,2,4,5,6,7>"""
import sys
import torch

src, dst, pi = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
assert sorted(pi) == list(range(8)) and src != dst, (pi, src, dst)


def reorder(t):
    b = list(t.reshape(8, t.shape[0] // 8, *t.shape[1:]))
    return torch.cat([b[pi.index(j)] for j in range(8)], 0).contiguous()


ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
sd = ck["state_dict"]
shape = tuple(sd["denoiser.final_layer.linear.weight"].shape)
for k in ("denoiser.final_layer.linear.weight", "ema_denoiser.final_layer.linear.weight"):
    sd[k] = reorder(sd[k])
n = 0
for o in ck.get("optimizer_states", []):
    hits = [i for i, st in o["state"].items() if torch.is_tensor(st.get("exp_avg")) and tuple(st["exp_avg"].shape) == shape]
    assert len(hits) == 1, f"expected one optimizer state of shape {shape}, found {hits}"
    st = o["state"][hits[0]]
    for m in ("exp_avg", "exp_avg_sq"):
        st[m] = reorder(st[m])
    n += 1
ck["final_layer_reordered"] = {"pi": pi, "source": src, "optimizer_states": n}
torch.save(ck, dst + ".tmp")
import os
os.replace(dst + ".tmp", dst)
print(f"wrote {dst}: final_layer weight (denoiser, EMA) and {n} optimizer state(s) re-ordered by pi {pi}")
