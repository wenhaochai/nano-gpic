#!/bin/bash
#SBATCH -J gpic-mfu-probe2
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever A (docs/mfu.md): FlashAttention-3 for the masked attention (GPIC_MFU_ATTN=fa3; HF kernels-hub build, owner OK
# 2026-11-06). Single-GPU tasks from a queue on all 8 GPUs (jobs/mfu/pool.sh), longest first:
# 1. GPU-only step time (tools/mfu/step_bench.py), SDPA / FA3 / SDPA on the same GPU, at seven widths;
# 2. loss parity: 300 steps at d640 with the same batches, t and noise: SDPA, FA3 twice, SDPA's math backend (the floor);
# 3. attention backends at the ladder's shapes, FA3 included (tools/mfu/attn_bench.py);
# 4. whole-model parity, SDPA and FA3 against fp32 (tools/mfu/test_attn_parity.py), eager and compiled.
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
source $WT/jobs/mfu/pool.sh
cd $WT
SB="timeout 900 python $WT/tools/mfu/step_bench.py"
aba() {   # one task line: SDPA, FA3, SDPA on one GPU
  echo "$SB --d $1 --steps 80 --out $OUT/d$1_sdpa1.json > $OUT/d$1_sdpa1.log 2>&1; GPIC_MFU_ATTN=fa3 $SB --d $1 --steps 80 --out $OUT/d$1_fa3.json > $OUT/d$1_fa3.log 2>&1; $SB --d $1 --steps 80 --out $OUT/d$1_sdpa2.json > $OUT/d$1_sdpa2.log 2>&1"
}
{
  aba 1792; aba 1536; aba 1280
  echo "GPIC_MFU_SDPA_BACKEND=MATH $SB --d 640 --steps 300 --record --out $OUT/loss_math.json > $OUT/loss_math.log 2>&1"
  aba 1024; aba 896; aba 768; aba 640
  echo "$SB --d 640 --steps 300 --record --out $OUT/loss_sdpa.json > $OUT/loss_sdpa.log 2>&1"
  echo "GPIC_MFU_ATTN=fa3 $SB --d 640 --steps 300 --record --out $OUT/loss_fa3.json > $OUT/loss_fa3.log 2>&1"
  echo "GPIC_MFU_ATTN=fa3 $SB --d 640 --steps 300 --record --out $OUT/loss_fa3_b.json > $OUT/loss_fa3_b.log 2>&1"
  echo "timeout 1200 python $WT/tools/mfu/attn_bench.py --out $OUT/attn.json > $OUT/attn.log 2>&1"
  echo "for a in '--d 640' '--d 640 --compile' '--d 1280 --nb 2 --nt 1' '--d 1792 --nb 2 --nt 1 --T 168'; do timeout 600 python $WT/tools/mfu/test_attn_parity.py \$a >> $OUT/parity.log 2>&1; echo \"parity \$a rc=\$?\" >> $OUT/parity.log; done"
} > $OUT/tasks.txt
pool $OUT/tasks.txt $OUT
grep -h "MFU\|PARITY" $OUT/*.log
echo "DONE $OUT"
