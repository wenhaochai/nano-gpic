#!/usr/bin/env python3
"""Loss parity of production-path arms (jobs/mfu/arm.sh with GPIC_MFU_DET_DATA=1: the same batches in every arm).

Per step, the global loss is the mean of the 8 ranks' losses (src/callbacks/bench.py records each rank's), so an arm
that deals samples over the ranks (lever E) compares on the same 256 samples. Reported per arm against the reference
arm: the relative difference of the per-step global loss (mean over steps 1-100 and over the last 200 steps, and the
largest), the mean loss over the last 500 steps of both. Verdict: PASS when the candidate's mean last-200 difference to
the reference is within 3x the floor (the second reference arm's difference to the first), and its mean last-500 loss
within 3x the floor's spread.
usage: python parity_loss.py <reference arm> <second reference arm> <candidate arm>...
"""
import glob, json, os, statistics as st, sys


def global_loss(arm):
    files = sorted(glob.glob(os.path.join(arm, "bench", "steps_rank*.json")))
    per = {}
    for f in files:
        for r in json.load(open(f))["rows"]:
            if "loss" in r:
                per.setdefault(r["step"], []).append(r["loss"])
    return {s: st.mean(v) for s, v in per.items() if len(v) == len(files)}, len(files)


def cmp(a, b):
    steps = sorted(set(a) & set(b))
    rel = [abs(a[s] - b[s]) / abs(b[s]) for s in steps]
    ab = [abs(a[s] - b[s]) for s in steps]
    tail = steps[-500:]
    return {"n": len(steps), "first100": st.mean(rel[:100]), "last200": st.mean(rel[-200:]), "max": max(rel),
            "abs_last200": st.mean(ab[-200:]), "mean_a": st.mean(a[s] for s in tail), "mean_b": st.mean(b[s] for s in tail)}


def main():
    ref, ref2, *cands = sys.argv[1:]
    R, nr = global_loss(ref)
    floor = cmp(global_loss(ref2)[0], R)
    print(f"reference {os.path.basename(ref)} ({nr} ranks, {len(R)} steps); floor {os.path.basename(ref2)}: "
          f"rel diff first100 {floor['first100']:.2e} last200 {floor['last200']:.2e} max {floor['max']:.2e}, "
          f"mean |loss diff| last200 {floor['abs_last200']:.2e}; "
          f"mean loss last 500: {floor['mean_a']:.6f} vs {floor['mean_b']:.6f}")
    ok = True
    for c in cands:
        r = cmp(global_loss(c)[0], R)
        good = (r["last200"] <= 3 * max(floor["last200"], 1e-6)
                and abs(r["mean_a"] - r["mean_b"]) <= 3 * max(abs(floor["mean_a"] - floor["mean_b"]), 1e-6 * r["mean_b"]))
        ok &= good
        print(f"{os.path.basename(c):24s} {r['n']} steps: rel diff first100 {r['first100']:.2e} last200 {r['last200']:.2e} "
              f"max {r['max']:.2e}, mean |loss diff| last200 {r['abs_last200']:.2e}; mean loss last 500 {r['mean_a']:.6f} vs "
              f"{r['mean_b']:.6f}: {'PASS' if good else 'FAIL'}")
    print("PARITY " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
