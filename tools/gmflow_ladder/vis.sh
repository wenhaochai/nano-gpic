#!/bin/bash
# Runs tools/gmflow_ladder/run.py on a vis node's CPU: vis.sh <file relative to the repo> [args...]. Start it from a login node with ssh vis2 (or vis1).
WT=$(cd "$(dirname "$0")/../.." && pwd)
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
cd $WT && exec nice -n 19 /scratch/gpfs/GROUP/USER/envs/pixgen/bin/python -I tools/gmflow_ladder/run.py "$@"
