#!/bin/bash
#SBATCH -J gpic-resumetest
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Does a resume load the whole optimizer state into the sharded optimizer? (gpic-28; owner 2026-11-08 「可以」, 「批准」.) The FP8 branch runs
# showed that a fresh optimizer at step 10200 costs FD 847.5 against 802.7; the d16 FP8 arm resumed at 11900 and lost FD +9.9%. A full
# checkpoint holds every AdamW moment and Muon momentum at full shape (checked); here a copy of branch <tag>'s last.ckpt is resumed for 32
# steps with GPIC_DEBUG_OPT_STATE=1, which prints the optimizer state's sums of squares per group as training starts, and they are compared
# with the checkpoint's own (tools/opt_state_sums.py). The same settings as the branch (d16, FSDP2 G1, Muon, bf16, no EMA).
# usage: resume_test.sh <branch tag>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
TAG=${1:?branch tag}
SRC=$WC/runs/gpic-speed/muon_fp8branch/$TAG/exp_ladder_d1536
ROOT=$WC/runs/gpic-speed/muon_fp8branch/resumetest_$TAG
DST=$ROOT/exp_ladder_d1536
WT=$WC/project/gpic-wt/fp8_branch_run
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
[ -f $SRC/last.ckpt ] || { echo "no $SRC/last.ckpt"; exit 2; }
mkdir -p $DST
[ -f $DST/last.ckpt ] || cp $SRC/last.ckpt $SRC/shard_seed.txt $SRC/shards_started_rank*.txt $DST/
export GPIC_MFU_NO_EMA=1 NCCL_NVLS_ENABLE=0 GPIC_DEBUG_OPT_STATE=1
export GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=1 GPIC_MFU_MUON_ADD=1 GPIC_MFU_FP8=
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT LADDER_LAST_EVERY=100000
S=3424
bash $WT/jobs/train_ladder.sh d1536 1536 16 4 $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr 0.000179 --model.optimizer.init_args.muon_lr 6.8e-3 \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay 0 \
  --model.lr_scheduler.init_args.num_warmup_steps 34 --model.lr_scheduler.init_args.num_cooldown_steps 2720 --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size 128 --trainer.accumulate_grad_batches 1 --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..07999}.tar' --seed_everything 0 \
  --trainer.logger.init_args.name resumetest_${TAG} --trainer.logger.init_args.id resumetest-$TAG 2>&1 | tee $ROOT/train.log | grep -a -E "optstate|Restor|resuming|Traceback|Error"
source $WC/envs/pixgen/bin/activate   # w2322 ran this with the system python (no torch); its sums matched by hand on vis1
python $WT/tools/opt_state_sums.py $SRC/last.ckpt | tee $ROOT/ckpt_sums.txt
python - $ROOT/train.log $ROOT/ckpt_sums.txt <<'PY'
import re, sys
pat = re.compile(r"group (\d+): (\d+) params, (\d+) with state; sum of squares exp_avg (\S+) exp_avg_sq (\S+) momentum_buffer (\S+)")
got = {m.group(1): m.groups()[1:] for m in pat.finditer(open(sys.argv[1], errors="replace").read()) if "[optstate]" in m.string[max(0, m.start() - 20):m.start()]}
ref = {m.group(1): m.groups()[1:] for m in pat.finditer(open(sys.argv[2]).read())}
ok = bool(ref) and set(got) == set(ref)
for g in sorted(ref):
    r, s = ref[g], got.get(g)
    if not s:
        print(f"group {g}: no [optstate] line"); ok = False; continue
    rel = [abs(float(a) - float(b)) / max(abs(float(b)), 1e-30) for a, b in zip(s[2:], r[2:])]
    same = s[0] == r[0] and s[1] == r[1] and all(x < 1e-6 for x in rel)
    ok &= same
    print(f"group {g}: params {s[0]}/{r[0]}, with state {s[1]}/{r[1]}, relative differences {[f'{x:.1e}' for x in rel]} {'same' if same else 'DIFFERENT'}")
print("RESUME_TEST", "PASS" if ok else "FAIL")
PY
