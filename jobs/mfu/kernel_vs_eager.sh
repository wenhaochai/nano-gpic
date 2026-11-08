#!/bin/bash
#SBATCH -J gpic-mfu-kve
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:45:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# genmodel 2026-11-07 23:xx: the d1536 ladder checkpoints score ~0.25 held-out (eager) against ~0.067 at the other widths.
# tools/mfu/kernel_vs_eager.py on 4 GPUs side by side, one environment each, the same batch and checkpoints (d1536 13600,
# d1792 8992, d1280 22400; raw, EMA, and a random-init d1536): eager | K + P | FA3 | FA3 + K2 + K + P (the training path).
# usage: kernel_vs_eager.sh <code worktree> <tools worktree>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
CODE=${1:?code worktree} TOOLS=${2:?tools worktree}
export TMPDIR=$WC/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p $TMPDIR; trap 'rm -rf $TMPDIR' EXIT
export TRITON_CACHE_DIR=$TMPDIR/triton
source $WC/envs/pixgen/bin/activate
R=$WC/runs/gpic-speed/muon_full O=$WC/runs/gpic-speed/mfu/kve_h100_${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p $O
CK="$R/exp_ladder_d1536/ms-step=013600.ckpt $R/exp_ladder_d1792/ms-step=008992.ckpt $R/exp_ladder_d1280/ms-step=022400.ckpt"
cd $TOOLS
run() {   # gpu tag env...
  local g=$1 tag=$2; shift 2
  env CUDA_VISIBLE_DEVICES=$g "$@" python tools/mfu/kernel_vs_eager.py run $CODE $O/$tag.pt $CK > $O/$tag.log 2>&1 &
}
run 0 eager
run 1 kp GPIC_MFU_FUSED_NORM=1 GPIC_MFU_PATCHIFY=1
run 2 fa3 GPIC_MFU_ATTN=fa3
run 3 train GPIC_MFU_ATTN=fa3 GPIC_MFU_FUSED_QK=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_PATCHIFY=1
wait
for t in eager kp fa3 train; do echo "== $t"; grep -v -i warn $O/$t.log | tail -8; done
for t in kp fa3 train; do echo "== eager vs $t"; python tools/mfu/kernel_vs_eager.py compare $O/eager.pt $O/$t.pt; done
