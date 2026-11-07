#!/usr/bin/env python3
"""Checks of an FSDP2 (lever B) checkpoint against a DDP one from the same recipe (probe8): both load; the same
state_dict keys, shapes and dtypes (so the evaluation code, which reads ema_denoiser.*, works unchanged); the EMA moved
away from the raw weights and is finite; the same optimizer state layout (per-parameter exp_avg and exp_avg_sq); the
same global step; the resumed arms started at that step.
usage: python check_ckpt.py <fsdp last.ckpt> <ddp last.ckpt> <probe dir>"""
import glob, json, os, sys
import torch


def main():
    fs, dd, out = sys.argv[1], sys.argv[2], sys.argv[3]
    a, b = (torch.load(p, map_location="cpu", weights_only=False) for p in (fs, dd))
    ok = True
    sa, sb = a["state_dict"], b["state_dict"]
    same_keys = sa.keys() == sb.keys()
    bad = [k for k in sa if k in sb and (sa[k].shape != sb[k].shape or sa[k].dtype != sb[k].dtype)]
    print(f"state_dict: {len(sa)} vs {len(sb)} keys, same keys {same_keys}, shape/dtype mismatches {bad[:5]}")
    ok &= same_keys and not bad
    ema = [k for k in sa if k.startswith("ema_denoiser.")]
    moved = sum(not torch.equal(sa[k], sa["denoiser." + k[len("ema_denoiser."):]]) for k in ema if "pos_embed" not in k)
    finite = all(torch.isfinite(sa[k]).all() for k in ema)
    print(f"EMA: {len(ema)} tensors, {moved} differ from the raw weights, all finite {finite}")
    ok &= finite and moved > 0.9 * len(ema) - 2
    for name, c in (("fsdp", a), ("ddp", b)):
        st = c["optimizer_states"][0]["state"]
        print(f"{name}: global_step {c['global_step']}, optimizer state entries {len(st)}, "
              f"keys {sorted(next(iter(st.values())).keys())}")
    ok &= a["global_step"] == b["global_step"]
    ka = sorted(tuple(v["exp_avg"].shape) for v in a["optimizer_states"][0]["state"].values())
    kb = sorted(tuple(v["exp_avg"].shape) for v in b["optimizer_states"][0]["state"].values())
    print(f"optimizer exp_avg shapes equal as multisets: {ka == kb}")
    ok &= ka == kb
    for arm in ("fsdp_b", "ddp_b"):
        f = sorted(glob.glob(os.path.join(out, arm, "bench", "steps_rank0.json")))
        if f:
            first = json.load(open(f[0]))["rows"][0]["step"]
            print(f"{arm}: first step {first}")
            ok &= first == a["global_step"]
        else:
            print(f"{arm}: no step timer output")
            ok = False
    print("CHECK " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
