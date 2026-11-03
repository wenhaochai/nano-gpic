#!/bin/bash
#SBATCH -J gpic-encode-captions
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The text encoder's outputs for every GPIC Lite training shard (tools/encode_captions.py, docs/text_cache.md), one
# process per GPU, every 8th shard each. A rerun skips finished shards, so a failed or stopped job is requeued as is.
# Exit 0 only when tools/check_text_cache.py finds every shard complete.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${CACHE_WT:-$WC/project/gpic-wt/ladder_cache}
OUT=$WC/data/gpic/train_emb_qwen3_1.7b
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
pids=()
for r in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$r python $WT/tools/encode_captions.py --shards "$WC/data/gpic/train/gpic_train_*.tar" --out $OUT \
    --rank $r --world 8 > $OUT/encode_rank$r.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
tail -qn 1 $OUT/encode_rank*.log
[ $fail -eq 0 ] || { echo "an encoder process failed"; exit 1; }
python $WT/tools/check_text_cache.py $OUT $WC/data/gpic/train
