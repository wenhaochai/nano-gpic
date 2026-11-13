#!/bin/bash
# Toy runs at the harder data, 6 at a time, one CPU thread each. usage: run_hard.sh <train steps>
cd /scratch/gpfs/GROUP/USER/tmp/dist_step
export CUDA_VISIBLE_DEVICES=
mkdir -p res2
for d in coupled manifold embedded; do
  for s in 0 1 2; do for a in point mix_t mix_multi; do echo "$d $a $s"; done; done
  echo "$d mix_nll 0"
done | nice -n 19 xargs -P 6 -L 1 bash -c '/scratch/gpfs/GROUP/USER/envs/pixgen/bin/python -I toy_head.py $0 $1 res2/$0_$1_s$2.json '"$1"' $2 >> res2/log.txt 2>&1'
echo done
