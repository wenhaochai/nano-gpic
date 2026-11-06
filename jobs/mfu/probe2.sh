#!/bin/bash
#SBATCH -J gpic-mfu-probe2
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever A (docs/mfu.md): FlashAttention-3 for the masked attention (GPIC_MFU_ATTN=fa3; HF kernels-hub build, owner OK
# 2026-11-06). One GPU per process, side by side:
# 1. attention backends at the ladder's shapes, FA3 included (tools/mfu/attn_bench.py);
# 2. whole-model parity, SDPA and FA3 against fp32 (tools/mfu/test_attn_parity.py), eager and compiled;
# 3. loss parity: 300 steps at d640 with the same batches, t and noise: SDPA, FA3, and SDPA's math backend (the floor);
# 4. GPU-only step time (tools/mfu/step_bench.py), SDPA / FA3 / SDPA on the same GPU, at seven widths.
# usage: probe2.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe2_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[ ${#G[@]} -eq 8 ] || { echo "need 8 GPUs, have ${CUDA_VISIBLE_DEVICES:-}"; exit 2; }
export TORCHINDUCTOR_CPP_WRAPPER=1 PIXGEN_SRC=$WT/baselines/PixelGen
cd $WT
sb() {   # sb <gpu> <tag> <d> [step_bench args...]; env from the caller
  local gpu=$1 tag=$2 d=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu timeout 900 python tools/mfu/step_bench.py --d $d --out $OUT/$tag.json "$@" > $OUT/$tag.log 2>&1
  echo "[$(date +%T)] $tag rc=$?"
}
aba() {   # aba <gpu> <d>: SDPA, FA3, SDPA
  sb $1 d$2_sdpa1 $2 --steps 80
  GPIC_MFU_ATTN=fa3 sb $1 d$2_fa3 $2 --steps 80
  sb $1 d$2_sdpa2 $2 --steps 80
}
( CUDA_VISIBLE_DEVICES=${G[0]} timeout 1200 python tools/mfu/attn_bench.py --out $OUT/attn.json > $OUT/attn.log 2>&1; echo "attn rc=$?"; aba ${G[0]} 768 ) &
( for args in "--d 640" "--d 640 --compile" "--d 1280 --nb 2 --nt 1" "--d 1792 --nb 2 --nt 1 --T 168"; do
    CUDA_VISIBLE_DEVICES=${G[1]} timeout 600 python tools/mfu/test_attn_parity.py $args >> $OUT/parity.log 2>&1; echo "parity $args rc=$?" | tee -a $OUT/parity.log
  done; aba ${G[1]} 896 ) &
( sb ${G[2]} loss_sdpa 640 --steps 300 --record --T hist; aba ${G[2]} 640 ) &
( GPIC_MFU_ATTN=fa3 sb ${G[3]} loss_fa3 640 --steps 300 --record --T hist; aba ${G[3]} 1024 ) &
( GPIC_MFU_SDPA_BACKEND=MATH sb ${G[4]} loss_math 640 --steps 300 --record --T hist; GPIC_MFU_ATTN=fa3 sb ${G[4]} loss_fa3_b 640 --steps 300 --record --T hist ) &
aba ${G[5]} 1280 &
aba ${G[6]} 1536 &
aba ${G[7]} 1792 &
wait
grep -h "MFU\|PARITY" $OUT/*.log
echo "DONE $OUT"
