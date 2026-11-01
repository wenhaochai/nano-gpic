#!/bin/bash
#SBATCH -J gpic-sample-eval
#SBATCH --gres=gpu:8
#SBATCH --time=02:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# Official GPIC evaluation: 50K test captions -> PixelGen predict (8 GPUs, 50 Euler steps, CFG 4.0, EMA net)
# -> gpic-eval FD-DINOv2/PRDC/MMD vs reference_stats/test_stats.npz. Phases are timed.
# usage: sample_eval.sh <run_dir with *.ckpt (latest is used) or a .ckpt> <out_dir>
set -eo pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
SRC=$1; OUT=$2
if [ -d "$SRC" ]; then CKPT=$(ls -t $(find "$SRC" -name "*.ckpt") | head -1); else CKPT=$SRC; fi
mkdir -p "$OUT"; echo "ckpt=$CKPT" | tee $OUT/ckpt.txt
cd $PIXGEN
t0=$(date +%s)
WANDB_MODE=disabled python main.py predict --config configs_h100/lite_sample.yaml --trainer.devices 8 \
  --trainer.default_root_dir "$OUT/pred" --ckpt_path "$CKPT" > $OUT/predict.log 2>&1
t1=$(date +%s)
IMG=$(dirname $(find "$OUT/pred" -name "*.png" | head -1))
echo "images in $IMG: $(ls $IMG | wc -l)"
cd $OUT
gpic-eval "$IMG" /scratch/gpfs/GROUP/USER/data/gpic/reference_stats/test_stats.npz \
  --batch_size 512 --num_workers 32 --device cuda:0 > $OUT/eval.log 2>&1
t2=$(date +%s)
echo "{\"predict_s\": $((t1-t0)), \"eval_s\": $((t2-t1))}" | tee $OUT/phase_times.json
grep -A12 "Metric" $OUT/eval.log | tail -12
# keep a compact copy of the images, drop the 50K PNG files (file-count quota)
python - "$IMG" "$OUT/samples_uint8.npz" <<'EOF'
import sys, os, numpy as np
from PIL import Image
d = sys.argv[1]; fs = sorted(os.listdir(d))
np.savez(sys.argv[2], arr_0=np.stack([np.asarray(Image.open(os.path.join(d, f)).convert("RGB")) for f in fs]), names=np.array(fs))
EOF
rm -rf "$IMG"
