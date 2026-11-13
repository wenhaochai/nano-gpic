#!/bin/bash
# Runs a Python command of this worktree on a vis node, on CPU: run_vis.sh <script relative to the repo root> [args...]. Start it with ssh vis1|2.
WT=$(cd "$(dirname "$0")/../.." && pwd)
source /scratch/gpfs/GROUP/USER/envs/pixgen/bin/activate
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6
exec nice -n 19 python "$WT/$1" "${@:2}"
