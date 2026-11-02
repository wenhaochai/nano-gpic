#!/bin/bash
# Publish the main branch to the public page-linked repo wenhaochai/generating-distributions: fresh clone,
# cluster paths and user names replaced in every commit (git-filter-repo), every blob scanned for secrets, then pushed.
# Run on vis1 (the scan runs Python). Prints the old -> new commit map for record links.
# usage: tools/publish_public.sh [branch=main]
set -euo pipefail
BR=${1:-main}
T=/scratch/gpfs/GROUP/USER/tmp/gd_public
SRC=/scratch/gpfs/GROUP/USER/project/gpic
FR=/scratch/gpfs/GROUP/USER/tmp/pls_blog/.venv/bin/git-filter-repo
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
git push -q https://github.com/wenhaochai/generating-distributions.git $BR:$BR
tail -5 .git/filter-repo/commit-map
