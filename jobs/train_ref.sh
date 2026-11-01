#!/bin/bash
#SBATCH -J gpic-lite-ref-e1
#SBATCH --gres=gpu:8
#SBATCH --time=13:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# Reference run with the ORIGINAL code: official GPIC PixelGen config, GPIC Lite (shards 0-800), 40k steps
# (~1 Lite epoch), official 400k lr schedule. Resumable: last.ckpt every 1000 steps; exit 0 only when done.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
RUN=$WC/runs/gpic-speed/exp_lite_ref_e1
n=$(ls $WC/data/gpic/train/gpic_train_00*.tar 2>/dev/null | wc -l)
if [ "$n" -lt 801 ]; then echo "only $n/801 Lite shards present; waiting"; exit 98; fi
CKPT_ARGS=()
if [ -f "$RUN/last.ckpt" ]; then CKPT_ARGS=(--ckpt_path "$RUN/last.ckpt"); echo "resuming from $RUN/last.ckpt"; fi
cd $PIXGEN
python main.py fit --config configs_h100/lite_ref_e1.yaml --trainer.devices 8 "${CKPT_ARGS[@]}"
rc=$?
echo "main.py exit $rc"
exit $rc
