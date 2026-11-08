#!/usr/bin/env python3
"""The optimizer state a full checkpoint holds, per param group: params with state and, per state key, the sum of squares over every
param (fp64), in the format of LightningModel._debug_opt_state's [optstate] lines (gpic-28 2026-11-08, resume test).
usage: opt_state_sums.py <last.ckpt>"""
import sys
import torch
o = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["optimizer_states"][0]
for gi, g in enumerate(o["param_groups"]):
    sums, n = [0.0, 0.0, 0.0], 0
    for i in g["params"]:
        st = o["state"].get(i, {})
        n += 1 if st else 0
        for j, k in enumerate(("exp_avg", "exp_avg_sq", "momentum_buffer")):
            v = st.get(k)
            if v is not None:
                sums[j] += v.double().pow(2).sum().item()
    print(f"[ckpt] group {gi}: {len(g['params'])} params, {n} with state; sum of squares exp_avg {sums[0]:.9e} "
          f"exp_avg_sq {sums[1]:.9e} momentum_buffer {sums[2]:.9e}", flush=True)
