#!/bin/bash
#SBATCH -J gpic-toy-chan
#SBATCH --account=group
#SBATCH --qos=short
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o /scratch/gpfs/GROUP/USER/tmp/dist_step/toy_chan_%j.out
# The arms of toy_code.py, one CPU thread each. A rerun skips the runs that finished.
[ -n "$SLURM_JOB_ID" ] || { echo "no job id"; exit 2; }
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export CUDA_VISIBLE_DEVICES=
cd /scratch/gpfs/GROUP/USER/tmp/dist_step || exit 2
mkdir -p res6
PY=/scratch/gpfs/GROUP/USER/envs/pixgen/bin/python
ARMS="code_chan_late_r1 code_chan_late_r2 code_chan_late_r4 code_chan_late_r8 code_chan_late_r16 code_chan_in_r4"
for s in 0 1 2; do for d in coupled manifold embedded; do for a in $ARMS; do echo "$d $a $s"; done; done; done \
  | PY=$PY xargs -P 32 -L 1 bash -c '[ -e res6/$0_$1_s$2.done ] || { $PY -I toy_code.py $0 $1 res6/$0_$1_s$2.json 10000 $2 > res6/$0_$1_s$2.log 2>&1 && touch res6/$0_$1_s$2.done; }'
echo "done $(ls res6/*.done | wc -l) runs"
