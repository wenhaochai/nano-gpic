#!/bin/bash
#SBATCH -J gpic-mfu-ncclverify
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# nccl-new-verify (owner popup 2026-11-08 11:2x): the gate before the ladder switches to NCCL 2.27.7 with NVLS on (w2148:
# 1-2% faster than NVLS off). NCCL 2.26.2 (torch 2.7.1) misorders ranks' blocks in NVLS all-gather and reduce-scatter at
# 576 KB per rank (NVIDIA/nccl #1906), the size of the d1536 final layer's shard, so every d1536 checkpoint save permuted
# it; 2.27.7 was right in the plain tests (w2147). Here the d1536 training path (debug-finalperm code: FSDP2 G 1, 128 x 1,
# 32 steps from scratch) runs with 2.27.7 preloaded and NVLS on, with the checks of w2130: forward vs shards at steps 2-31,
# the call-time gathers, and the explicit all-gather vs full_tensor (the save path) from step 2 (steps 0-1 hold the
# zero-initialized final layer). It passes only if every check is the identity and the log shows NCCL 2.27.7 loaded on
# every rank and NVLS buffers set up.
# usage: nccl_new_verify.sh <debug-finalperm code worktree>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
WT=${1:?debug-finalperm code worktree}
LIB=$WC/tmp/nccl_wheels/2.27.7/nvidia/nccl/lib/libnccl.so.2
[ -f $LIB ] || { echo "no $LIB"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q GPIC_DEBUG_FINAL_PERM_RANKS $WT/baselines/PixelGen/src/lightning_model_gpic.py || { echo "$WT has no final-layer checks"; exit 2; }
PAT='gpic_train_{00000..07999}.tar'
BASE="GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=1 GPIC_MFU_MUON_ADD=1"
ROOT=$WC/runs/gpic-speed/debug_finalperm/ncclnew_${FETCHED_JOB_ID:-$SLURM_JOB_ID}
LOG=$ROOT/train.log
mkdir -p $ROOT/exp_ladder_d1536; echo 20261106 > $ROOT/exp_ladder_d1536/shard_seed.txt
env $BASE LD_PRELOAD=$LIB NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,NVLS GPIC_DEBUG_FINAL_PERM_RANKS=1 WANDB_MODE=offline \
  GPIC_DEBUG_FINAL_PERM=1 GPIC_DEBUG_FINAL_PERM_EVERY=1 LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT LADDER_LAST_EVERY=32 \
  bash $WT/jobs/train_ladder.sh d1536 1536 16 4 32 \
    --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr 1.8e-4 --model.optimizer.init_args.muon_lr 6.8e-3 \
    --data.train_batch_size 128 --trainer.accumulate_grad_batches 1 --data.train_dataset.init_args.shared_bucket true \
    --data.train_dataset.init_args.tar_pattern "$PAT" \
    --trainer.logger.init_args.name debug_ncclnew --trainer.logger.init_args.id debug-ncclnew > $LOG 2>&1
echo "training exit $?"
grep -a -o -E "NCCL version [0-9.]+\+cuda[0-9.]+" $LOG | sort | uniq -c
grep -a -o -E "\[finalperm\] step [0-9]+: (forward blocks|explicit all-gather)[^;]*" $LOG | head -8
ID="[0, 1, 2, 3, 4, 5, 6, 7]"
FW="\[finalperm\] step ([2-9]|[12][0-9]|3[01]): forward blocks <- storage blocks: weight \[[0-9, -]*\]"
CALL="at the final layer's call: unsharded weight blocks <- shards gathered now \[[0-9, -]*\]"
EX="\[finalperm\] step ([2-9]|[1-9][0-9]+): explicit all-gather blocks <- full_tensor blocks \[[0-9, -]*\]"
n_fw=$(grep -a -o -E "$FW" $LOG | wc -l); bad_fw=$(grep -a -o -E "$FW" $LOG | grep -v -F "weight $ID" | wc -l)
n_call=$(grep -a -o -E "$CALL" $LOG | wc -l); bad_call=$(grep -a -o -E "$CALL" $LOG | grep -v -F "now $ID" | wc -l)
n_ex=$(grep -a -o -E "$EX" $LOG | wc -l); bad_ex=$(grep -a -o -E "$EX" $LOG | grep -v -F "blocks $ID" | wc -l)
n_ver=$(grep -a -c -E "NCCL version 2\.27\.7\+" $LOG); n_old=$(grep -a -c -E "NCCL version 2\.26\." $LOG)
n_nvls=$(grep -a -c -E "NVLS rank [0-7] \(dev [0-7]\) alloc done" $LOG)
echo "VERIFY: forward-vs-shards checks $n_fw (wrong $bad_fw), call-time gathers $n_call (wrong $bad_call), explicit-vs-full_tensor $n_ex (wrong $bad_ex); NCCL 2.27.7 lines $n_ver, 2.26 lines $n_old; NVLS alloc lines $n_nvls"
if [ $n_fw -ge 25 ] && [ $bad_fw -eq 0 ] && [ $n_call -ge 2 ] && [ $bad_call -eq 0 ] && [ $n_ex -ge 2 ] && [ $bad_ex -eq 0 ] \
   && [ $n_ver -ge 1 ] && [ $n_old -eq 0 ] && [ $n_nvls -ge 8 ]; then echo "NCCL_NEW_VERIFY PASS"; exit 0; fi
echo "NCCL_NEW_VERIFY FAIL"; exit 1
