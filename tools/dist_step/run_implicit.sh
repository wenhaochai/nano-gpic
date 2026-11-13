#!/bin/bash
# The implicit sampler arms, 6 at a time, one CPU thread each. usage: run_implicit.sh <compute in point-arm steps> <seeds> <arms...>
cd /scratch/gpfs/GROUP/USER/tmp/dist_step
export CUDA_VISIBLE_DEVICES=
mkdir -p res3
B=$1; SEEDS=$2; shift 2
for d in coupled manifold embedded; do
  for s in $SEEDS; do for a in "$@"; do echo "$d $a $s"; done; done
done | nice -n 19 xargs -P 6 -L 1 bash -c '[ -e res3/$0_$1_s$2.json ] || /scratch/gpfs/GROUP/USER/envs/pixgen/bin/python -I toy_implicit.py $0 $1 res3/$0_$1_s$2.json '"$B"' $2 >> res3/log.txt 2>&1'
echo done
