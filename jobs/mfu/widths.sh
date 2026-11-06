#!/bin/bash
#SBATCH -J gpic-mfu-widths
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU per-width table (docs/mfu.md): on the production path (jobs/mfu/arm.sh, real Lite batches, shard seed 20261106, the
# same batches in every arm, each rank's loss per step), for each given width: levers off, then each given lever set.
# usage: widths.sh <code worktree> <steps> <width>:<set>[,<set>...] ...   sets: ACD, ACDE, ABCD, ABCDE (A FA3, B FSDP2,
#   C GPU resize, D uint8, E packed + rebalanced captions, L AdamW and EMA on the local shards (B2), G<k> k blocks per
#   FSDP2 unit, P torch.profiler over steps 500-505 and per-stage CUDA events: read the median), off (a second
#   levers-off arm, d<width>_off2)
# Parity floor (tools/mfu/parity_loss.py): off2 if listed, else the first set (put a rounding-only set first: ACD changes
# only the attention kernel's rounding, C and D are bit-exact).
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree} STEPS=${2:?steps}; shift 2
OUT=$WC/runs/gpic-speed/mfu/widths_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PIXGEN_SRC=$WT/baselines/PixelGen
source $WT/jobs/mfu/gpu_checks.sh
gpu_checks $WT $OUT || { echo "GPU checks failed"; exit 3; }
export MFU_WT=$WT MFU_LIMIT=900 GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106
FSDP='{"class_path":"lightning.pytorch.strategies.ModelParallelStrategy","init_args":{"data_parallel_size":8,"tensor_parallel_size":1,"save_distributed_checkpoint":false}}'
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; echo "[$(date +%T)] $1 rc=$?" | tee -a $OUT/summary.txt; }
geom() { case $1 in 640) echo "9 2" ;; 768) echo "10 2" ;; 896) echo "11 2" ;; 1024) echo "12 3" ;; 1280) echo "14 3" ;; 1536) echo "16 4" ;; 1792) echo "18 4" ;; esac; }
for spec in "$@"; do
  d=${spec%%:*}; sets=${spec#*:}
  read -r nb nt <<< "$(geom $d)"
  arm d${d}_off $d $nb $nt $STEPS
  for s in ${sets//,/ }; do
    [ $s = off ] && { arm d${d}_off2 $d $nb $nt $STEPS; continue; }
    env=(GPIC_MFU_ATTN=fa3)
    [[ $s == *B* ]] && env+=(GPIC_MFU_FSDP=1 MFU_STRATEGY=$FSDP)
    [[ $s == *C* ]] && env+=(GPIC_MFU_GPU_RESIZE=1)
    [[ $s == *D* ]] && env+=(GPIC_MFU_UINT8=1)
    [[ $s == *E* ]] && env+=(GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=1)
    [[ $s == *L* ]] && env+=(GPIC_MFU_LOCAL_OPT=1)
    [[ $s =~ G([0-9]+) ]] && env+=(GPIC_MFU_FSDP_GROUP=${BASH_REMATCH[1]})
    [[ $s == *P* ]] && env+=(BENCH_STAGES=1 BENCH_PROFILE=500:505)
    ( export "${env[@]}"; arm d${d}_$s $d $nb $nt $STEPS )
  done
done
cd $WT
python tools/mfu/bench_summary.py $(ls -d $OUT/d*/) --skip 100 | tee -a $OUT/summary.txt
for spec in "$@"; do
  d=${spec%%:*}; sets=${spec#*:}
  arms=(); for s in ${sets//,/ }; do [ $s = off ] && arms=($OUT/d${d}_off2 "${arms[@]}") || arms+=($OUT/d${d}_$s); done
  python tools/mfu/parity_loss.py $OUT/d${d}_off "${arms[@]}" | tee -a $OUT/summary.txt
done
echo "DONE $OUT"
