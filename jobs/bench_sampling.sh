#!/bin/bash
#SBATCH -J gpic-bench-sampling
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Sampling throughput on 8 H100s (docs/gmflow_hn.md, N2; tools/gmflow/bench_sampling.py): images / s of the samplers alone (no FD) at C0/8 of d640: the point model w1147
# (official Euler at 50, 16, 8, 4 calls) and the K = 8 model w1173 (GM-ODE at 8 and 4 calls, PyTorch sub-steps and the fused Triton kernel). bf16, compiled, after a
# warm-up pass, at least 60 s per setting on 1024 prompts per GPU. Checks fused = not fused in fp32 (per call at the first and the last call; end to end at 8 and 4
# calls on 32 prompts per GPU; max |diff| < 1e-4); exit 1 if a check fails, after the JSON is written. The bf16 end-to-end difference is a report only.
# Expected: about 20 min. Output: runs/gpic-speed/gmflow_k8_d640/exp_ladder_d640/bench_sampling_<date>.json (next to the K = 8 run) and the log beside it.
# Code: the frozen worktree gpic-wt/gmflow_run3 (the merge of gmflow-hn into gmflow, 2026-11-06).
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/gmflow_run3
[ -f "$WT/baselines/PixelGen/src/diffusion/flow_matching/gm_ode_triton.py" ] || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
POINT=$WC/runs/gpic-speed/point_nodrop_d640/exp_ladder_d640/ms-step=066816.ckpt
GM=$WC/runs/gpic-speed/gmflow_k8_d640/exp_ladder_d640/ms-step=062336.ckpt
[ -f $POINT ] && [ -f $GM ] || { echo "missing checkpoint"; exit 2; }
OUT=$WC/runs/gpic-speed/gmflow_k8_d640/exp_ladder_d640/bench_sampling_$(date +%m%d_%H%M)
echo "code=$(git -C $WT rev-parse HEAD)" > $OUT.commit.txt
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled OMP_NUM_THREADS=8
timeout -k 60 2700 python -m torch.distributed.run --standalone --nproc_per_node 8 $WT/tools/gmflow/bench_sampling.py --point $POINT --gm $GM --out $OUT.json \
  --prompts 1024 --min-seconds 60 > $OUT.log 2>&1
rc=$?
tail -n 15 $OUT.log
echo "bench rc=$rc $OUT.json"
exit $rc
