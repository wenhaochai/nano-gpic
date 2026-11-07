#!/bin/bash
#SBATCH -J gpic-bench8-c8
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# C8 (DDP comm hook averaging inside NCCL) against C7 on the official 8-GPU shape, same node, 120 steps each.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
G=/scratch/gpfs/GROUP/USER/project/gpic-wt
R=$G/perf-opt/workspace-runs/c8_$(date +%m%d_%H%M)
mkdir -p $R
python $G/perf-opt/tools/gpu_health.py > $R/gpu_health.txt 2>&1 || { cat $R/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cat $R/gpu_health.txt
log() { echo "[$(date +%T)] $*" | tee -a $R/phases.log; }
run() { local c=$1 tag=$2; log "START $tag"; env BENCH_DEVICES=8 BENCH_STEPS=120 TORCHINDUCTOR_CACHE_DIR=$R/cache/$tag/inductor \
  TRITON_CACHE_DIR=$R/cache/$tag/triton bash $G/cand/$c/jobs/bench_train.sh $R/$tag; log "END $tag rc=$?"; }
run C7 C7_8
run C8 C8_8
run C7 C7_8b
case "$R" in */workspace-runs/c8_*) rm -rf -- "$R/cache";; esac
log "DONE $R"
