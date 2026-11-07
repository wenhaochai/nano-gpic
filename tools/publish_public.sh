#!/bin/bash
# Publish the project to the public page-linked repo wenhaochai/nano-gpic (owner 2026-11-07: 「commit为什么不全啊，然后main要
# 创建一个吧，然后要全吧」, 「公开仓库不要提及h100」, 「说H100就行」): every branch at once, each with its whole history; the
# production line prod (main with ladder, mfu and muon-search2 merged) becomes the public main. In every commit, file name
# and message: cluster paths and user names replaced, the cluster's name replaced by H100 (configs_h100 -> configs_h100,
# node names generic); then every blob, message and ref name scanned (secrets, cluster ids, the cluster's name); then a forced
# push (the rewrite changes every hash). Prints the old -> new commit map for the page's Code links.
# Run on a vis node (the scan runs Python). The public README is rewritten at every publish (owner 2026-11-07: 「readme每次commit
# 都重写一下说明一下我们的ladder搭建和目标 和已经实现的东西 也要适当配图」): refuse unless README.md on prod was committed today.
# usage: tools/publish_public.sh [--dry]   (--dry: everything but the push)
set -euo pipefail
DRY=${1:-}
T=/scratch/gpfs/GROUP/USER/tmp/gd_public
SRC=/scratch/gpfs/GROUP/USER/project/gpic
FR=/scratch/gpfs/GROUP/USER/tmp/pls_blog/.venv/bin/git-filter-repo
URL=https://github.com/wenhaochai/nano-gpic.git
SCAN=$(cd "$(dirname "$0")" && pwd)/pub_scan.py   # this script's own scanner (the cluster's name included)
# private branch -> public branch; main itself is not published (its content is in main)
MAP=(prod:main ar:ar dist-step:dist-step energy-head:energy-head eval-par:eval-par gmflow:gmflow heldout-loss:heldout-loss
     implicit-nodrop:implicit-nodrop ladder:ladder mfu:mfu muon:muon muon-search2:muon-search2)
git -C $SRC fetch -q origin
RD=$(git -C $SRC log -1 --format=%cs origin/prod -- README.md)
[ "$RD" = "$(date +%F)" ] || { echo "README.md on prod was last rewritten on $RD: rewrite it (the ladder, the goals, what works, figures) first"; exit 1; }
mkdir -p $T && cd $T
case "$T/repo" in */tmp/gd_public/repo) rm -rf -- "$T/repo";; esac
git init -q repo && cd repo
PUB=()
for m in "${MAP[@]}"; do
  src=${m%%:*}; dst=${m#*:}
  git fetch -q --no-tags $SRC "refs/remotes/origin/$src:refs/heads/$dst"
  PUB+=("$dst")
done
SUB='
import re
REPL = [(b"/scratch/gpfs/GROUP/USER", b"/scratch/gpfs/GROUP/USER"), (b"USER", b"USER"), (b"GROUP", b"GROUP"),
        (b"group", b"group"), (b"USER", b"USER"),
        (b"configs_h100", b"configs_h100"), (b"main", b"main"), (b"vis1", b"vis1"), (b"vis2", b"vis2"),
        (b"proxy", b"proxy"), (b"login2", b"login2"), (b"login1", b"login1")]
NODE = re.compile(rb"h100-([a-z]+[0-9][0-9a-z]*)")
def sub(b):
    for a, c in REPL: b = b.replace(a, c)
    b = NODE.sub(rb"node-\1", b)
    return b.replace(b"H100", b"H100").replace(b"H100", b"H100").replace(b"h100", b"h100")
'
"$FR" --force --refs "${PUB[@]}" --message-callback "$SUB
return sub(message)" --filename-callback "$SUB
return sub(filename)" --blob-callback "$SUB
d = blob.data
if b'\x00' not in d[:8000]:
    blob.data = sub(d)
"
fail=0
for b in "${PUB[@]}"; do
  python3 -I $SCAN $T/repo $b > $T/scan_$b.txt
  if grep -qE "^== (wandb|hf_token|openai|anthropic|google|github|slack|aws|private_key|overleaf|bearer|cluster_ids)" $T/scan_$b.txt; then
    echo "$b: the scan found secrets or cluster ids:"; grep -A3 "^== " $T/scan_$b.txt | head -20; fail=1
  fi
done
if git log --format=%B "${PUB[@]}" | grep -qi h100 || git for-each-ref --format='%(refname)' | grep -qi h100; then
  echo "a commit message or a ref name still names the cluster"; fail=1
fi
[ $fail = 0 ] || { echo "not pushing"; exit 1; }
for b in "${PUB[@]}"; do echo "$b: $(git rev-list --count $b) commits"; done
if [ "$DRY" = --dry ]; then echo "dry run: nothing pushed"; exit 0; fi
git push -q --force $URL "${PUB[@]/#/refs/heads/}"
echo "pushed ${#PUB[@]} branches; old -> new commit map: $T/repo/.git/filter-repo/commit-map"
tail -3 .git/filter-repo/commit-map
