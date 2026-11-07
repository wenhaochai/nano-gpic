#!/bin/bash
#SBATCH -J gpic-heldout-smoke
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:30:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of tools/heldout_loss.py on H100 (submit with wq --prio 10): four checkpoints, Lite and Full d640 at ms-step=133600
# and the vis2-checked Lite 133600 / Full 534400 pair plus Lite / Full 267200, N = 256 + 256, on 4 GPUs (a whole wq block) through the shared
# task list (one worker per GPU), with the extra
# split train_reenc (train images, captions encoded on the held-out path). Checks the JSON lines (fields, 16 levels, the
# mean of the levels, EMA != raw, train_reenc within 0.5% of train), compares them with the same checkpoints on a vis2
# A40 (tmp/heldout_loss/vischeck.jsonl, 2026-11-07), then measures seconds per forward at d640 and d1792 (bench) for
# the time estimate. Output: tmp/heldout_loss/smoke_<job>.jsonl. Writes nothing into runs/.
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_loss
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
JID=${FETCHED_JOB_ID:-${SLURM_JOB_ID:-manual}}
OUT=$WC/tmp/heldout_loss/smoke_$JID.jsonl
R=$WC/runs/gpic-speed
CK=($R/exp_ladder_d640/ms-step=133600.ckpt $R/full_ladder/exp_ladder_d640/ms-step=534400.ckpt $R/exp_ladder_d640/ms-step=267200.ckpt $R/full_ladder/exp_ladder_d640/ms-step=267200.ckpt)
rm -f $OUT
t0=$(date +%s)
HELDOUT_SPLITS=val,train,train_reenc heldout_run $OUT 256 256 4 "${CK[@]}"
echo "[smoke] four checkpoints on 4 GPUs: $(( $(date +%s) - t0 )) s"
python - $OUT $WC/tmp/heldout_loss/vischeck.jsonl "${CK[@]}" <<'PY'
import json, sys
out, ref, cks = sys.argv[1], sys.argv[2], sys.argv[3:]
recs = [json.loads(l) for l in open(out)]
assert sorted((r["ckpt"], r["weights"]) for r in recs) == sorted((c, w) for c in cks for w in ("ema", "raw")), recs
need = ("run_root", "ladder", "size", "d", "nb", "nt", "step", "train_flops", "weights", "n_val", "n_train", "num_t",
        "loss_val", "loss_train", "loss_val_t", "loss_train_t", "t", "elapsed_s")
for r in recs:
    miss = [k for k in need if k not in r]
    assert not miss, miss
    assert r["num_t"] == 16 and len(r["loss_val_t"]) == 16 and len(r["loss_train_t"]) == 16 and len(r["t"]) == 16
    assert abs(sum(r["loss_val_t"]) / 16 - r["loss_val"]) < 1e-9 and 0.01 < r["loss_val"] < 0.5, r["loss_val"]
    assert abs(r["loss_train_reenc"] / r["loss_train"] - 1) < 5e-3, (r["loss_train_reenc"], r["loss_train"])
    assert (r["d"], r["nb"], r["nt"], r["n_val"], r["n_train"]) == (640, 9, 2, 256, 256)
    print(f"[smoke] {r['ladder']} {r['size']} {r['step']} {r['weights']}: val {r['loss_val']:.5f} train {r['loss_train']:.5f} "
          f"train_reenc {r['loss_train_reenc']:.5f}; {r['elapsed_s']} s on {r['gpu']}")
by = {(r["ckpt"], r["weights"]): r for r in recs}
for c in cks:
    assert by[(c, "ema")]["loss_val"] != by[(c, "raw")]["loss_val"]
try:
    vis = {(r["ckpt"], r["weights"]): r for r in map(json.loads, open(ref))}
    for k, r in by.items():
        if k in vis:
            d = max(abs(r[f] / vis[k][f] - 1) for f in ("loss_val", "loss_train"))
            print(f"[smoke] {k[1]} {r['ladder']} {r['step']}: relative difference to the vis2 A40 {d:.2e}")
            assert d < 5e-3, d
except FileNotFoundError:
    print("[smoke] no vis2 reference")
print("[smoke] JSON lines OK")
PY
for a in "--d 640 --nb 9 --nt 2" "--d 1024 --nb 12 --nt 3" "--d 1792 --nb 18 --nt 4"; do
  python $WT/tools/heldout_loss.py bench $a --iters 10 --n-val 3072 --n-train 3072 2>&1 | grep "^{"
done
echo "[smoke] PASS"
