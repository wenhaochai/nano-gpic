#!/bin/bash
#SBATCH -J gpic-smoke-ckptonelast
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke of the checkpoint hooks of ckpt-onelast-cap / ckpt-onelast-sched before they go to production (gpic-28 2026-11-09):
# no save at the step a launch resumed from, and last.ckpt always the live state (src/callbacks/model_checkpoint.py; the
# CPU test is tools/mfu/tests/test_ckpt_resume_save.py). The two cases that left -vN files in production, through the
# real launcher on 8 GPUs, about 30 min in all:
#  A an extension at width 1792 (FSDP2, 2 micro-batches a step: the muon_full run that wrote ms-step=008992-v1.ckpt):
#    96 steps (ms every 24, last every 48), then max_steps 192 (ms every 48, last every 96), so the launch resumes at a
#    step that both new intervals divide. New code (ckpt-onelast-cap): launch 2 saves only steps 144 and 192. Old code
#    (ckpt-dedup-cap), the same two launches: it must show the fault (ms-step=000096-v1.ckpt, last-v1.ckpt), or the
#    launches test nothing.
#  B a branch at width 640 (DDP, 1 micro-batch a step) on ckpt-onelast-sched: a parent of 1024 steps, then a new run
#    directory with a copy of the parent's last.ckpt and shard log, max_steps 2048 with both intervals changed: last.ckpt
#    is written over (before: last-v1.ckpt beside a stale last.ckpt), and the parent's directory stays as it was.
# tools/mfu/check_ckpt_onelast.py reads each launch's "[ckpt]" lines and the files.
# Result: SMOKE_CKPT_ONELAST PASS | FAIL <reasons>; the checkpoints are deleted on PASS and kept on FAIL.
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
G=$WC/project/gpic-wt
NC=$G/onelast_cap_run NS=$G/onelast_sched_run OLD=$G/ckptdedup_cap_run
HOOK=baselines/PixelGen/src/callbacks/model_checkpoint.py
for w in $NC $NS $OLD; do
  [ -z "$(git -C $w status --porcelain --untracked-files=no)" ] || { echo "$w has local changes"; exit 2; }
  echo "code $(basename $w) $(git -C $w rev-parse --short HEAD)"
done
cmp -s $NC/$HOOK $NS/$HOOK || { echo "the hook differs between $NC and $NS"; exit 2; }
grep -q "micro-batches" $NC/$HOOK && ! grep -q "micro-batches" $OLD/$HOOK || { echo "$NC must hold the new hook and $OLD the old one"; exit 2; }
OUT=$WC/runs/gpic-speed/mfu_probe/ckptonelast_$(date +%m%d_%H%M); mkdir -p $OUT
MAN=$OUT/manifest.txt; : > $MAN
FAIL=""
run() {   # <worktree> <tag> <d> <nb> <nt> <max steps> <last every> <mb> <fsdp> <schedule steps>: one launch of the sweep path
  local WT=$1 tag=$2 d=$3 nb=$4 nt=$5 S=$6 LAST=$7 MB=$8 FS=$9 SCH=${10} root=$OUT/$2 size=d$3
  local WU=$((SCH / 100)); [ $WU -ge 1 ] || WU=1
  mkdir -p $root/exp_ladder_$size; [ -s $root/exp_ladder_$size/shard_seed.txt ] || echo 20261106 > $root/exp_ladder_$size/shard_seed.txt
  GPIC_MFU_NO_EMA=1 LD_PRELOAD=$WC/envs/nccl/2.27.7/libnccl.so.2 \
  GPIC_MFU_FSDP=$FS GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=$FS \
  GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=$FS GPIC_MFU_MUON_ADD=1 LADDER_WT=$WT LADDER_RUN_ROOT=$root LADDER_LAST_EVERY=$LAST BENCH_OUT=$root/bench_$(date +%s) \
  exec bash $WT/jobs/train_ladder.sh $size $d $nb $nt $S \
    --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr 0.00064 --model.optimizer.init_args.muon_lr 1.36e-2 \
    --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay 0.002752 \
    --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $(( (SCH / 5) / 32 * 32 )) \
    --model.lr_scheduler.init_args.max_steps $SCH --model.lr_scheduler.init_args.min_lr 3e-5 \
    --data.train_batch_size $MB --trainer.accumulate_grad_batches $((1024 / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
    --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..07999}.tar' \
    --trainer.logger.init_args.name smoke_ckptonelast_${tag} --trainer.logger.init_args.id smoke-ckptonelast-$tag-$(date +%m%d%H%M)
}
launch() {   # <tag> <launch number> <worktree> <d> <nb> <nt> <max steps> <last every> <mb> <fsdp> <schedule steps>; one schedule for all launches of a run
  local tag=$1 n=$2 wt=$3; shift 3
  echo "== $tag launch $n on $(basename $wt): max_steps $4, last.ckpt every $5 ($(date +%T))"
  local t0=$(date +%s)
  ( export -f run; export OUT WC; timeout --kill-after=120 900s bash -c "run $wt $tag $*" ) > $OUT/$tag.$n.log 2>&1
  local rc=$?
  echo "$tag launch $n: rc $rc in $(( $(date +%s) - t0 )) s; $(grep -a -h 'resuming from' $OUT/$tag.$n.log | head -1)"
  grep -a -o '\[ckpt\] step.*' $OUT/$tag.$n.log | sed "s/^/$tag launch $n: /"
  [ $rc = 0 ] || FAIL="$FAIL $tag-launch$n-rc$rc"
}
list() {   # <dir> <name pattern>: name, size, mtime, inode of each file
  find $1 -maxdepth 1 -type f -name "$2" -printf '%f %s %T@ %i\n' | sort
}

A=exp_ladder_d1792
for c in a_new:$NC a_old:$OLD; do   # the training sources env.sh itself (it owns and removes the job's TMPDIR, as in a sweep job)
  tag=${c%%:*} wt=${c#*:}
  launch $tag 1 $wt 1792 18 4 96 48 64 1 192
  list $OUT/$tag/$A "ms-step=*.ckpt" > $OUT/$tag.before
  launch $tag 2 $wt 1792 18 4 192 96 64 1 192
  list $OUT/$tag/$A "ms-step=*.ckpt" > $OUT/$tag.after
  ls -l --time-style=+%T $OUT/$tag/$A/*.ckpt | awk -v t=$tag '{print t ": " $5, $6, $7}' | sed "s|$OUT/$tag/$A/||"
done
cat >> $MAN <<EOF
launch a_new.1 $OUT/a_new.1.log 0 96 24 48 2
launch a_new.2 $OUT/a_new.2.log 96 192 48 96 2
files a_new $OUT/a_new/$A 192 24,48,72,96,144,192
same a_new-ms $OUT/a_new.before $OUT/a_new.after kept
fault a_old $OUT/a_old/$A 96
EOF

B=exp_ladder_d640
launch b_parent 1 $NS 640 9 2 1024 128 128 0 2048
mkdir -p $OUT/b_branch/$B && cp $OUT/b_parent/$B/last.ckpt $OUT/b_parent/$B/shards_started_rank*.txt $OUT/b_parent/$B/shard_seed.txt $OUT/b_branch/$B/ || FAIL="$FAIL b-copy"
list $OUT/b_parent/$B "*" > $OUT/b_parent.before
launch b_branch 1 $NS 640 9 2 2048 256 128 0 2048
list $OUT/b_parent/$B "*" > $OUT/b_parent.after
ls -l --time-style=+%T $OUT/b_branch/$B/*.ckpt | awk '{print "b_branch: " $5, $6, $7}' | sed "s|$OUT/b_branch/$B/||"
cat >> $MAN <<EOF
launch b_parent.1 $OUT/b_parent.1.log 0 1024 256 128 1
files b_parent $OUT/b_parent/$B 1024 256,512,768,1024
launch b_branch.1 $OUT/b_branch.1.log 1024 2048 512 256 1
files b_branch $OUT/b_branch/$B 2048 1536,2048
same b_parent $OUT/b_parent.before $OUT/b_parent.after exact
EOF

source $WC/project/gpic/jobs/env.sh
echo "== check ($(date +%T))"
OMP_NUM_THREADS=16 python $NC/tools/mfu/check_ckpt_onelast.py $MAN 2>&1 | tee $OUT/check.log
[ ${PIPESTATUS[0]} = 0 ] || FAIL="$FAIL check"

if [ -z "$FAIL" ]; then
  find $OUT -name "*.ckpt" -delete
  echo "SMOKE_CKPT_ONELAST PASS $OUT"
else
  echo "SMOKE_CKPT_ONELAST FAIL$FAIL $OUT (checkpoints kept: $(du -sh $OUT | cut -f1))"
  exit 1
fi
