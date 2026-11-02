#!/bin/bash
#SBATCH -J gpic-bench8-c7
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:55:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# The official 8-GPU shape: base and C7 at 32 images per GPU (global 256), a profile window of C7, then on one node side
# by side 4-GPU C7 at 64 images per GPU (global 256) and at 32 (global 128), for the cost per image.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
G=/scratch/gpfs/GROUP/USER/project/gpic-wt
R=$G/perf-opt/workspace-runs/bench8_$(date +%m%d_%H%M)
mkdir -p $R
GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); [ ${#GPUS[@]} -ge 8 ] || GPUS=(0 1 2 3 4 5 6 7)
A="${GPUS[0]},${GPUS[1]},${GPUS[2]},${GPUS[3]}"; B="${GPUS[4]},${GPUS[5]},${GPUS[6]},${GPUS[7]}"
log() { echo "[$(date +%T)] $*" | tee -a $R/phases.log; }
run() { local tag=$1; shift; log "START $tag"; env TORCHINDUCTOR_CACHE_DIR=$R/cache/$tag/inductor TRITON_CACHE_DIR=$R/cache/$tag/triton "$@"; log "END $tag rc=$?"; }
nvidia-smi --query-gpu=index,clocks.sm,power.limit,temperature.gpu,clocks_event_reasons.active --format=csv > $R/gpus_start.csv
run base8 env BENCH_DEVICES=8 bash $G/cand/base/jobs/bench_train.sh $R/base8
run C7_8 env BENCH_DEVICES=8 bash $G/cand/C7/jobs/bench_train.sh $R/C7_8
run prof_C7_8 env BENCH_DEVICES=8 bash $G/cand/C7/jobs/bench_train.sh $R/prof_C7_8 100:103
run C7_4_b64 env CUDA_VISIBLE_DEVICES=$A MASTER_PORT=29521 BENCH_DEVICES=4 BENCH_STEPS=100 BENCH_ARGS="--data.train_batch_size 64" bash $G/cand/C7/jobs/bench_train.sh $R/C7_4_b64 &
run C7_4_b32 env CUDA_VISIBLE_DEVICES=$B MASTER_PORT=29522 BENCH_DEVICES=4 BENCH_STEPS=100 bash $G/cand/C7/jobs/bench_train.sh $R/C7_4_b32 &
wait
nvidia-smi --query-gpu=index,clocks.sm,power.limit,temperature.gpu,clocks_event_reasons.active --format=csv > $R/gpus_end.csv
log "DONE $R"
