#!/bin/bash
# The tangent heads at the harder data. usage: run_tangent.sh <train steps>
cd /scratch/gpfs/GROUP/USER/tmp/dist_step
export CUDA_VISIBLE_DEVICES=
for d in embedded manifold coupled; do
  for s in 0 1 2; do for a in tangent tangent_mix; do echo "$d $a $s"; done; done
done | nice -n 19 xargs -P 6 -L 1 bash -c '/scratch/gpfs/GROUP/USER/envs/pixgen/bin/python -I toy_head.py $0 $1 res2/$0_$1_s$2.json '"$1"' $2 >> res2/log.txt 2>&1'
echo done
