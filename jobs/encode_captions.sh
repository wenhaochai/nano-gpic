#!/bin/bash
#SBATCH -J gpic-encode-captions
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=06:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# The text encoder's outputs for every GPIC training shard in data/gpic/train (tools/encode_captions.py of the frozen
# worktree gpic-wt/ladder_cache, docs/text_cache.md), one process per GPU, every 8th shard each. A rerun skips finished
# shards, so a failed or stopped job is requeued as is. Exit 0 only when tools/check_text_cache.py finds every shard complete.
# GPIC Full (owner 2026-11-06: the full cache for the full ladder): 8000 shards, the cache grows from 1.66 TiB (Lite's 801
# shards) to about 16.6 TiB, about 36 GPU-h. The fileset is shared, so the job starts writing only when the free space
# plus what is already written is at least 20 TiB, which leaves about 3.4 TiB free at the end (the Lite guard was 3.5 TiB
# for 1.66 TiB). Otherwise exit 98: wq puts it back in the queue for 30 min.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${CACHE_WT:-$WC/project/gpic-wt/ladder_cache}
OUT=$WC/data/gpic/train_emb_qwen3_1.7b
[ -d "$WT" ] || { echo "no worktree $WT"; exit 2; }
mkdir -p $OUT
# a process killed during a write leaves a temporary file (up to 2.1 GiB), and a rerun uses a new pid
rm -f "$OUT"/*.tmp*
free_g=$(df -BG --output=avail $WC | tail -1 | tr -d ' G')
written_g=$(du -s --apparent-size -BG $OUT | cut -f1 | tr -d G)
echo "free ${free_g} GiB, already written ${written_g} GiB"
[ $((free_g + written_g)) -ge 20480 ] || { echo "less than 20 TiB free plus written: waiting"; exit 98; }
git -C $WT rev-parse HEAD > $OUT/commit.txt
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
# logical GPU r -> the id this job was given (a host may hand out any block of its GPUs)
GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); [ ${#GPUS[@]} -eq 8 ] || { echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES is not 8 GPUs"; exit 98; }
pids=()
for r in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=${GPUS[$r]} python $WT/tools/encode_captions.py --shards "$WC/data/gpic/train/gpic_train_*.tar" --out $OUT \
    --rank $r --world 8 > $OUT/encode_rank$r.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
tail -qn 1 $OUT/encode_rank*.log
[ $fail -eq 0 ] || { echo "an encoder process failed"; exit 1; }
python $WT/tools/check_text_cache.py $OUT $WC/data/gpic/train
