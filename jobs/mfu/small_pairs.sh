#!/bin/bash
#SBATCH -J gpic-mfu-smallp
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:10:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Two small exact levers as pairs on the GPU halves (jobs/mfu/pairs.sh, ladder setting with P and NS): the EMA on the shards'
# local tensors (GPIC_MFU_LOCAL_OPT=1, FSDP2 sizes d17 and d20), then Muon's updates in one launch (GPIC_MFU_MUON_ADD=1; d15
# DDP and d20 FSDP2; its GPU test first). usage: small_pairs.sh <code worktree>
set -o pipefail
WT=${1:?code worktree}
bash $WT/jobs/mfu/pairs.sh $WT localema "GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1" "GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=1" none "1280 1024" "1536 768"
bash $WT/jobs/mfu/pairs.sh $WT muonadd "GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1" "GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1 GPIC_MFU_MUON_ADD=1" test_muon_add "1024 1024" "1536 768"
