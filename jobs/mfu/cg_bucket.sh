#!/bin/bash
#SBATCH -J gpic-mfu-cgbucket
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:55:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# cudagraph-bucket (owner 2026-11-08 12:2x 「可以」, popup 12:3x): lever X (the compiled denoiser replayed as CUDA graphs,
# GPIC_MFU_CUDAGRAPH=1 + GPIC_PROF_DEN_COMPILE_MODE=reduce-overhead + Inductor's Python launcher) again, now with the shared
# bucket: every micro-batch pads to one of the 10 bucket edges, so the graphs are recorded once per edge (CompileWarmup
# records them all before the first step) instead of once per caption length (w1470: new lengths kept coming, the mean
# step got worse). At the DDP widths, where the host's launches can bind (X gave d896 -6.4% median in w1470).
# 1. smoke: 128 steps at d640 on both halves (A without, B with X); stop unless B ran with finite losses and only bucket
#    lengths. 2. jobs/mfu/pairs.sh at d640, d768, d896 (A = the sweep's DDP switches, B = A + X), then
#    tools/mfu/pair_window.py (the halves compared while both train in steady state). The same kernels in the same order:
#    B's losses should match A's as far as a no-op pair's do (about 20-30 steps) and agree over the last 200 steps.
# usage: cg_bucket.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
EA="GPIC_MFU_NS_FUSED=1 GPIC_MFU_MUON_ADD=1 GPIC_MFU_NO_EMA=1 NCCL_NVLS_ENABLE=0"
EB="$EA GPIC_MFU_CUDAGRAPH=1 GPIC_PROF_DEN_COMPILE_MODE=reduce-overhead TORCHINDUCTOR_CPP_WRAPPER=0"
SM=$WC/runs/gpic-speed/mfu/cgbucket_smoke_$(date +%m%d_%H%M)
mkdir -p $SM
python $WT/tools/gpu_health.py > $SM/gpu_health.txt 2>&1 || { tail -3 $SM/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
smoke() {   # <gpus> <port> <env> <tag>
  (
    export PIXGEN_SRC=$WT/baselines/PixelGen MFU_WT=$WT GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106 MFU_LIMIT=900 MFU_MICRO=128
    export GPIC_MFU_ATTN=fa3 GPIC_MFU_GPU_RESIZE=1 GPIC_MFU_UINT8=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_SHARED_BUCKET=1
    export CUDA_VISIBLE_DEVICES=$1 MASTER_PORT=$2 $3
    bash $WT/jobs/mfu/arm.sh $SM $4 640 9 2 128 --config configs_h100/muon_ladder.yaml --trainer.devices 4
    echo "[$(date +%T)] $4 rc=$?" | tee -a $SM/summary.txt
  )
}
smoke 0,1,2,3 29531 "$EA" smokeA &
TORCHINDUCTOR_CACHE_DIR=$WC/cache/inductor/gpic_pairB TRITON_CACHE_DIR=$WC/cache/triton/gpic_pairB smoke 4,5,6,7 29532 "$EB" smokeB &
wait
grep -a -h "\[warmup\]" $SM/smokeB/train.log | tail -4 | tee -a $SM/summary.txt
python - $SM <<'PY' | tee -a $SM/summary.txt
import json, math, os, sys
sm = sys.argv[1]; edges = {16, 24, 32, 40, 48, 56, 64, 144, 176, 320}
for tag in ("smokeA", "smokeB"):
    f = os.path.join(sm, tag, "bench", "steps_rank0.json")
    r = json.load(open(f))["rows"] if os.path.exists(f) else []
    Ts = sorted({x.get("T") for x in r})
    ok = len(r) >= 64 and all(math.isfinite(x["loss"]) for x in r) and set(Ts) <= edges
    print(f"{tag}: {len(r)} timed steps, lengths {Ts}, finite losses {all(math.isfinite(x['loss']) for x in r)}: {'OK' if ok else 'BAD'}")
PY
grep -q "smokeB: .*OK" $SM/summary.txt || { echo "CG_BUCKET smoke failed (see $SM/smokeB/train.log)"; grep -a -h -E "Error|Traceback" $SM/smokeB/train.log | head -5; exit 1; }
bash $WT/jobs/mfu/pairs.sh $WT cgbucket "$EA" "$EB" none "640 3072" "768 2048" "896 2048"
python $WT/tools/mfu/pair_window.py $(ls -d $WC/runs/gpic-speed/mfu/pairs_cgbucket_* | tail -1)
