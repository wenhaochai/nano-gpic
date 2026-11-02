#!/bin/bash
#SBATCH -J gpic-bench8-nccl
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# C7 on 8 GPUs (official shape) under NCCL settings that take fewer SMs from the backward GEMMs: the default (to log what
# NCCL picks at 8 GPUs), 12 ring channels, NVLS only, and NVLS for all-reduce with ring for broadcast. NCCL INFO/TUNING
# logs show the algorithm actually used. 100 steps each.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
G=/scratch/gpfs/GROUP/USER/project/gpic-wt
R=$G/perf-opt/workspace-runs/nccl8_$(date +%m%d_%H%M)
mkdir -p $R
log() { echo "[$(date +%T)] $*" | tee -a $R/phases.log; }
run() { local tag=$1; shift; log "START $tag"; env BENCH_DEVICES=8 BENCH_STEPS=100 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING \
  TORCHINDUCTOR_CACHE_DIR=$R/cache/inductor TRITON_CACHE_DIR=$R/cache/triton "$@" bash $G/cand/C7/jobs/bench_train.sh $R/$tag; log "END $tag rc=$?"; }
run default
run ring12 NCCL_MAX_NCHANNELS=12
run nvls NCCL_ALGO=NVLS
run nvls_ar "NCCL_ALGO=allreduce:nvls;broadcast:ring"
log "DONE $R"
