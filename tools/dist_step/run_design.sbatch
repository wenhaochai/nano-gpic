#!/bin/bash
#SBATCH -J gpic-toy-design
#SBATCH --account=group
#SBATCH --qos=short
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH -o /scratch/gpfs/GROUP/USER/tmp/dist_step/toy_design_%j.out
# The toy arms of toy_design.py, one CPU thread each. A rerun skips the runs that finished.
[ -n "$SLURM_JOB_ID" ] || { echo "no job id"; exit 2; }
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export CUDA_VISIBLE_DEVICES=
cd /scratch/gpfs/GROUP/USER/tmp/dist_step || exit 2
mkdir -p res4
PY=/scratch/gpfs/GROUP/USER/envs/pixgen/bin/python
ARMS="point impl_late_m2 cvae_once_l0.01 cvae_once_l0.1 cvae_once_l1 cvae_step_l0.01 cvae_step_l0.1 cvae_step_l1 part_k8 gauss_r0 gauss_r0d gauss_r4 nll_d nll_r4 flow_w48"
for s in 0 1 2; do for d in coupled manifold embedded; do for a in $ARMS; do echo "$d $a $s"; done; done; done \
  | PY=$PY xargs -P 32 -L 1 bash -c '[ -e res4/$0_$1_s$2.done ] || { $PY -I toy_design.py $0 $1 res4/$0_$1_s$2.json 10000 $2 > res4/$0_$1_s$2.log 2>&1 && touch res4/$0_$1_s$2.done; }'
echo "done $(ls res4/*.done | wc -l) runs"
