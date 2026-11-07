#!/bin/bash
#SBATCH -J gpic-mfu-fp8p34
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# FP8 phases 3 and 4 after the triton-op fix (owner popup 2026-11-07 18:1x): the extended FP8 GPU test (compiled = eager with
# the C++ wrapper; phase 4's casts and block) gates the run; then two pairs on the GPU halves (jobs/mfu/pairs.sh, ladder
# setting, lever P in both): native2 against native3, and native3 against native4, at d1024 and d1536.
# usage: fp8_pairs2.sh <code worktree>
set -o pipefail
WT=${1:?code worktree}
bash $WT/jobs/mfu/pairs.sh $WT fp8n23 "GPIC_MFU_PATCHIFY=1 GPIC_MFU_FP8=native2" "GPIC_MFU_PATCHIFY=1 GPIC_MFU_FP8=native3" test_fp8_native "1024 1024" "1536 768" || exit $?
bash $WT/jobs/mfu/pairs.sh $WT fp8n34 "GPIC_MFU_PATCHIFY=1 GPIC_MFU_FP8=native3" "GPIC_MFU_PATCHIFY=1 GPIC_MFU_FP8=native4" none "1024 1024" "1536 768"
