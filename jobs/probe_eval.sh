#!/bin/bash
#SBATCH -J gpic-mfu-probe-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# P-eval of the optimization scan (gpic-28 2026-11-09; owner popup 「两个都跑」), on eval-probe (= eval_official_run9 + fast_eval.py
# --gpu-resize + EVAL_PRED_BATCH) and heldout-probe (GPIC_HELDOUT_COMPILE=1):
#  E1 d9 predict, default compile against mode reduce-overhead (CUDA graphs): loop time and the samples' md5
#  E2 d9 predict at 64 per GPU (not bit for bit by design): loop time
#  E3 d16 predict, 48 batches, default against reduce-overhead: loop time and md5
#  E4 DINOv2 on E1's samples: the CPU transform against tools/dino_gpu_resize.py (exact copy on the GPU): metrics, features, feature_s
#  E5 held-out loss of d9, d10 and two d16 checkpoints, eager (GPUs 0-3) and compiled (GPUs 4-7) side by side: values and time
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
source $WC/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/eval_probe_run HWT=$WC/project/gpic-wt/heldout_probe_run
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] && [ -z "$(git -C $HWT status --porcelain --untracked-files=no)" ] || { echo "probe worktrees have local changes"; exit 2; }
RV=$WC/runs/gpic-speed/raw_view
OUT=$WC/runs/gpic-speed/mfu_probe/eval_$(date +%m%d_%H%M); mkdir -p $OUT
SHM=$(mktemp -d /dev/shm/gpic_probe_XXXXXX); trap 'rm -rf -- "$SHM"' EXIT
link() {   # <name> <run under raw_view> <size> <step>
  local d=$OUT/$1/exp_ladder_$3; mkdir -p $d
  ln -sf $RV/$2/exp_ladder_$3/ms-step=$(printf %06d $4).ckpt $d/
}
pred() {   # <name> <size> <d> <nb> <nt> <step> [env...]: a predict only (EVAL_PHASE=predict), samples in $SHM; prints loop time and md5
  local name=$1 size=$2 d=$3 nb=$4 nt=$5 step=$6; shift 6
  env "$@" EVAL_PHASE=predict EVAL_SHM_ROOT=$SHM GMFLOW_WT=$WT LADDER_RUN_ROOT=$OUT/$name EVAL_SIZE=$size EVAL_D=$d EVAL_NB=$nb EVAL_NT=$nt \
    EVAL_SPREAD=image EVAL_NOISE=official bash $WT/jobs/eval_gmflow.sh 0 $step euler 50 > $OUT/$name.log 2>&1
  local o=$(ls -d $OUT/$name/exp_ladder_$size/eval_* | head -1)
  local npy=$(cat $o/.npz_path 2>/dev/null || find $(ls -td $SHM/task_* | head -1) -name "output.np[yz]" | head -1)   # LIMIT: no .npz_path
  local loop=$(tr '\r' '\n' < $o/predict.log | grep -a -o -E "[0-9]+/[0-9]+ [0-9:]+ • [0-9:-]+ [0-9.]+it/s" | tail -1)
  echo "$name: loop [$loop] md5 $(md5sum < $npy | cut -c1-16) samples $npy" | tee -a $OUT/summary.txt
}
link e1def muon_sweep/x-d9-c1-lr3p40e-3 d640 133600
link e1cg muon_sweep/x-d9-c1-lr3p40e-3 d640 133600
link e2b64 muon_sweep/x-d9-c1-lr3p40e-3 d640 133600
link e3def muon_fp8recipe/fp8-d20-bf16 d1536 13600
link e3cg muon_fp8recipe/fp8-d20-bf16 d1536 13600
echo "== E1, E2, E3 $(date +%T)" | tee -a $OUT/summary.txt
pred e1def d640 640 9 2 133600
pred e1cg d640 640 9 2 133600 GPIC_PROF_DEN_COMPILE_MODE=reduce-overhead TORCHINDUCTOR_CPP_WRAPPER=0
pred e2b64 d640 640 9 2 133600 EVAL_PRED_BATCH=64
pred e3def d1536 1536 16 4 13600 EVAL_LIMIT_BATCHES=48
pred e3cg d1536 1536 16 4 13600 EVAL_LIMIT_BATCHES=48 GPIC_PROF_DEN_COMPILE_MODE=reduce-overhead TORCHINDUCTOR_CPP_WRAPPER=0
echo "== E4 $(date +%T)" | tee -a $OUT/summary.txt
NPY=$(cat $(ls -d $OUT/e1def/exp_ladder_d640/eval_*)/.npz_path)
REF=$WC/data/gpic/reference_stats/test_stats.npz
for v in cpu gpu; do
  A=(); [ $v = cpu ] || A=(--gpu-resize)
  python -m torch.distributed.run --standalone --nproc_per_node 8 $WT/tools/fast_eval.py "$NPY" $REF $OUT/e4_$v.json --workers 6 "${A[@]}" \
    --save-features $OUT/e4_$v.npy > $OUT/e4_$v.log 2>&1
  echo "e4 $v rc=$?" | tee -a $OUT/summary.txt
done
python - $OUT <<'PY' | tee -a $OUT/summary.txt
import json, sys, numpy as np
o = sys.argv[1]
a, b = json.load(open(f"{o}/e4_cpu.json")), json.load(open(f"{o}/e4_gpu.json"))
fa, fb = np.load(f"{o}/e4_cpu.npy"), np.load(f"{o}/e4_gpu.npy")
print(f"E4: features bitwise equal {np.array_equal(fa, fb)} {fa.shape}; metrics equal {all(a[k] == b[k] for k in a if not k.endswith('_s'))}; "
      f"FD {a['fd']:.3f} / {b['fd']:.3f}; feature_s cpu {a['feature_s']:.1f} gpu {b['feature_s']:.1f}; total_s {a['total_s']:.1f} / {b['total_s']:.1f}")
PY
rm -f $OUT/e4_*.npy
echo "== E5 $(date +%T)" | tee -a $OUT/summary.txt
R=$WC/runs/gpic-speed
CK=($R/muon_sweep/x-d9-c1-lr3p40e-3/exp_ladder_d640/ms-step=133600.ckpt $RV/muon_sweep/x-d10-c4-lr2p72e-2/exp_ladder_d768/ms-step=021248.ckpt
    $R/muon_fp8recipe/fp8-d20-bf16/exp_ladder_d1536/ms-step=013600.ckpt $R/muon_fp8recipe/fp8-d20-n3-s2c/exp_ladder_d1536/ms-step=013600.ckpt)
( export WT=$HWT PYTHONPATH=$HWT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HELDOUT_TRAIN_SHARDS="00787"
  source $HWT/jobs/heldout_loss_lib.sh
  CUDA_VISIBLE_DEVICES=0,1,2,3 FETCHED_JOB_ID=probe_eager_$$ heldout_run $OUT/e5_eager.jsonl 3072 3072 4 "${CK[@]}" > $OUT/e5_eager.log 2>&1 &
  CUDA_VISIBLE_DEVICES=4,5,6,7 FETCHED_JOB_ID=probe_comp_$$ GPIC_HELDOUT_COMPILE=1 heldout_run $OUT/e5_comp.jsonl 3072 3072 4 "${CK[@]}" > $OUT/e5_comp.log 2>&1 &
  wait )
python - $OUT <<'PY' | tee -a $OUT/summary.txt
import json, sys
o = sys.argv[1]
rd = lambda f: {(r["ckpt"], r["weights"]): r for r in map(json.loads, open(f"{o}/{f}"))}
e, c = rd("e5_eager.jsonl"), rd("e5_comp.jsonl")
for k in sorted(e):
    if k[1] != "raw" or k not in c: continue
    a, b = e[k], c[k]
    print(f"E5 {k[0].split('/')[-3]:22s} {k[0].split('/')[-2]:14s}: val {a['loss_val']:.6f} / {b['loss_val']:.6f} (rel {b['loss_val'] / a['loss_val'] - 1:+.2e}); "
          f"train rel {b['loss_train'] / a['loss_train'] - 1:+.2e}; elapsed {a['elapsed_s']:.0f} s eager, {b['elapsed_s']:.0f} s compiled")
PY
find $OUT -name "*.ckpt" -type l -delete
echo PROBE_EVAL_DONE $OUT
