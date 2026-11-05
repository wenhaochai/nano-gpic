#!/bin/bash
#SBATCH -J gpic-ar-flow
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Does the head's flow in one compiled graph keep the FD and save time at evaluation? ar_few_d640 at step 52320, sampled as drawn. Reading, set before the run, in docs/autoregressive.md, section 采样速度. GPU utilization each second goes to gpu_util.csv. Expected: about 10 min. Code: the frozen worktree gpic-wt/ar_flow.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ar_flow
grep -q "GPIC_AR_HEAD_FLOW=1" $WT/jobs/eval_ar.sh || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/ar_few_d640
mkdir -p $LADDER_RUN_ROOT/exp_ladder_d640/eval_052320_ar_flow
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw --format=csv,noheader,nounits -l 1 > $LADDER_RUN_ROOT/exp_ladder_d640/eval_052320_ar_flow/gpu_util.csv &
MON=$!
bash $WT/jobs/eval_ar.sh d640 640 9 2 52320 flow > $LADDER_RUN_ROOT/eval_052320_flow.log 2>&1
rc=$?
kill $MON 2>/dev/null
echo "eval flow rc=$rc"
exit $rc
