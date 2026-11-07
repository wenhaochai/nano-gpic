#!/bin/bash
# Publish the main branch to the public page-linked repo wenhaochai/nano-gpic: fresh clone,
# cluster paths and user names replaced in every commit (git-filter-repo), every blob scanned for secrets, then pushed.
# Run on vis1 (the scan runs Python). Prints the old -> new commit map for record links.
# usage: tools/publish_public.sh [branch=main]
set -euo pipefail
BR=${1:-main}
T=/scratch/gpfs/GROUP/USER/tmp/gd_public
SRC=/scratch/gpfs/GROUP/USER/project/gpic
FR=/scratch/gpfs/GROUP/USER/tmp/pls_blog/.venv/bin/git-filter-repo
# The public README is rewritten at every publish (owner 2026-11-07: 「readme每次commit都重写一下说明一下我们的ladder搭建和目标 和已经实现的东西
# 也要适当配图」): refuse unless README.md on main (the default branch) was committed today; publish main with every branch.
RD=$(git -C $SRC log -1 --format=%cs main -- README.md)
[ "$RD" = "$(date +%F)" ] || { echo "README.md on main was last rewritten on $RD: rewrite it (the ladder, the goals, what works, figures) first"; exit 1; }
mkdir -p $T && cd $T
case "$T/repo" in */tmp/gd_public/repo) rm -rf -- "$T/repo";; esac
git clone -q --no-local --single-branch -b $BR $SRC repo && cd repo
SUB='
REPL = [(b"/scratch/gpfs/GROUP/USER", b"/scratch/gpfs/GROUP/USER"), (b"USER", b"USER"), (b"GROUP", b"GROUP"), (b"group", b"group")]
def sub(b):
    for a, c in REPL: b = b.replace(a, c)
    return b
'
"$FR" --force --refs $BR --message-callback "$SUB
return sub(message)" --filename-callback "$SUB
return sub(filename)" --blob-callback "$SUB
d = blob.data
if b'\x00' not in d[:8000]:
    blob.data = sub(d)
"
python3 -I $SRC/tools/pub_scan.py $T/repo $BR | tee $T/scan.txt
if grep -qE "^== (wandb|hf_token|openai|anthropic|google|github|slack|aws|private_key|overleaf|bearer|cluster_ids)" $T/scan.txt; then
  echo "scan found secrets or cluster ids: not pushing"; exit 1
fi
git push -q https://github.com/wenhaochai/nano-gpic.git $BR:$BR
tail -5 .git/filter-repo/commit-map
