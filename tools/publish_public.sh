#!/bin/bash
# Publish the project to the public page-linked repo wenhaochai/nano-gpic (a fresh repo since 2026-11-07, owner: 「wenhaochai/nano-gpic吧」; before it gpic-baseline; owner 2026-11-07: 「commit为什么不全啊，然后main要
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
# private branch -> public branch; main itself is not published (its content is in main); ladder-p is the code the
# ladder runs from the FSDP2 widths on (3338a8c plus lever P)
MAP=(prod:main ar:ar dist-step:dist-step energy-head:energy-head eval-par:eval-par gmflow:gmflow heldout-loss:heldout-loss
     implicit-nodrop:implicit-nodrop ladder:ladder ladder-p:ladder-p ladder-pns:ladder-pns mfu:mfu muon:muon muon-search2:muon-search2
     ladder-pnsr:ladder-pnsr ladder-pnsr-ma:ladder-pnsr-ma ladder-sweep:ladder-sweep sweep-noema:sweep-noema sweep-noema-cap:sweep-noema-cap
     ladder-noema:ladder-noema ladder-datamem:ladder-datamem eval-official:eval-official
     cooldown:cooldown ladder-full:ladder-full gmflow-b2:gmflow-b2 ladder-pns-fp8:ladder-pns-fp8
     ladder-sweep-fp8-cap:ladder-sweep-fp8-cap fp8-branch:fp8-branch sweep-noema-sched:sweep-noema-sched heldout-release:heldout-release
     bucket-sub:bucket-sub resume-roundtrip:resume-roundtrip
     eval-official-run9:eval-official-run9 heldout-loss-compiled:heldout-loss-compiled mfu-probe-small:mfu-probe-small
     mfu-probe-prof:mfu-probe-prof eval-probe:eval-probe heldout-probe:heldout-probe)   # 2026-11-08: the code of every run the page records; 2026-11-09 (gpic-28): the deployed eval (run9), the compiled held-out tool, the optimization scan's probes
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
# The GPIC authors out of the public Contributors (owner 2026-11-07: 「不想看到别人」, 「claude可以有，只是不要那三位原作者」): the
# GPIC release's 10 commits (b79f925..afa82da, keshik6/gpic's main when this fork started, by its authors) become one import commit
# by the owner that names the source, grafted under every branch; its metadata is fixed, so the rewrite stays deterministic. LICENSE
# and the README's credit stay as they are; the Co-Authored-By trailers stay. The checks refuse any other author or committer.
UP=afa82daab73cfeab4ef9fbd174ff62b68bffc456
UPS=" $(git rev-list $UP | tr '\n' ' ') "
IMP=$(GIT_AUTHOR_NAME="Wenhao Chai" GIT_AUTHOR_EMAIL=USER@gmail.com GIT_COMMITTER_NAME="Wenhao Chai" \
  GIT_COMMITTER_EMAIL=USER@gmail.com GIT_AUTHOR_DATE="$(git log -1 --format=%aI $UP)" GIT_COMMITTER_DATE="$(git log -1 --format=%cI $UP)" \
  git commit-tree "$UP^{tree}" -m "Import the GPIC code release" \
  -m "The code of github.com/keshik6/gpic at afa82da (GPIC; Chandrasegaran, Sargent et al., 2026) as one commit. LICENSE and the original README (GPIC_README.md) are kept.")
git rev-list --parents "${PUB[@]}" --not $UP | while read -r c ps; do
  np=(); hit=0
  for p in $ps; do case "$UPS" in *" $p "*) np+=("$IMP"); hit=1;; *) np+=("$p");; esac; done
  [ $hit = 0 ] || git replace --graft "$c" "${np[@]}"
done
SUB='
import re
REPL = [(b"wenhaochai/nano-gpic", b"wenhaochai/nano-gpic"), (b"/scratch/gpfs/GROUP/USER", b"/scratch/gpfs/GROUP/USER"), (b"USER", b"USER"), (b"GROUP", b"GROUP"),
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
m = sub(message).replace(b'; links moved to the rewritten public hashes', b'')
if m.startswith(b'publish_public: the GPIC release'):   # the owner's quotes stay private (owner 2026-11-07)
    m = b'publish_public: the code release as one import commit; refuse any author or committer other than the owner' + m[m.index(b'\n'):]
return m" --filename-callback "$SUB
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
if git log --format='%an <%ae>%n%cn <%ce>' "${PUB[@]}" | sort -u | grep -vxF 'Wenhao Chai <USER@gmail.com>' | grep -q .; then
  echo "a commit by someone other than the owner:"; git log --format='%an <%ae>%n%cn <%ce>' "${PUB[@]}" | sort | uniq -c; fail=1
fi
for b in "${PUB[@]}"; do [ "$(git rev-list --max-parents=0 $b | wc -l)" = 1 ] && [ "$(git log -1 --format=%s $(git rev-list --max-parents=0 $b))" = "Import the GPIC code release" ] || { echo "$b: the root is not the import commit"; fail=1; }; done
[ $fail = 0 ] || { echo "not pushing"; exit 1; }
for b in "${PUB[@]}"; do echo "$b: $(git rev-list --count $b) commits"; done
if [ "$DRY" = --dry ]; then echo "dry run: nothing pushed"; exit 0; fi
git push -q --force $URL "${PUB[@]/#/refs/heads/}"
echo "pushed ${#PUB[@]} branches; old -> new commit map: $T/repo/.git/filter-repo/commit-map"
tail -3 .git/filter-repo/commit-map
