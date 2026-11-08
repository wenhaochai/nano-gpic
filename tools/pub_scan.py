"""Secret/privacy scan of every blob reachable from the given refs of a repo (rules from tmp/ssov_scan/pub_scan.py). Masked output.
usage: pub_scan.py <repo> [ref ...]   (all refs when none)"""
import re, subprocess, sys, collections, tempfile
repo = sys.argv[1]
RULES = {
    "wandb_key_assign": re.compile(r"(?i)wandb[_ ]?api[_ ]?key\s*[=:]\s*['\"]?([0-9a-f]{40})"),
    "wandb_netrc": re.compile(r"machine\s+api\.wandb\.ai.*?password\s+([0-9a-f]{40})", re.S),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "google_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "overleaf_token": re.compile(r"\bolp_[A-Za-z0-9]{20,}|git\.overleaf\.com/[0-9a-f]{24}|://[^/\s:@]+:[^@\s/]{8,}@"),
    "password_assign": re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[=:]\s*['\"]?([^\s'\"]{8,})"),
    "bearer": re.compile(r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]{20,}"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "cluster_ids": re.compile(r"USER|GROUP|group|USER|[Dd]ella|H100"),   # the cluster's name too (owner 2026-11-07: 公开仓库不要提及h100)
}
# every object reachable from the refs given (all refs when none), each blob read once through one `git cat-file --batch`
# (2026-11-08: two git processes per object, once per published branch, took ~8 min; owner: 「推送为什么那么慢」)
refs = sys.argv[2:] or ["--all"]
objs = subprocess.run(["git", "-C", repo, "rev-list", "--objects", *refs], capture_output=True, text=True, check=True).stdout.splitlines()
paths = {}
for line in objs:
    sha, _, path = line.partition(" ")
    if path:
        paths.setdefault(sha, path)
chk = subprocess.run(["git", "-C", repo, "cat-file", "--batch-check=%(objectname) %(objecttype)"], input="".join(x + "\n" for x in paths),
                     capture_output=True, text=True, check=True).stdout.split()
blob_shas = [chk[i] for i in range(0, len(chk) - 1, 2) if chk[i + 1] == "blob"]
# the requests come from a file, not a pipe: writing them all into a pipe before reading deadlocks once git's output fills its own
# pipe (2026-11-08: a scan of 26 branches sat 4 h at 0% CPU)
req = tempfile.TemporaryFile()
req.write("".join(x + "\n" for x in blob_shas).encode()); req.seek(0)
proc = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"], stdin=req, stdout=subprocess.PIPE)
hits = collections.defaultdict(set)
n = 0
for sha in blob_shas:
    head = proc.stdout.readline().split()
    size = int(head[2]); data = proc.stdout.read(size); proc.stdout.read(1)
    assert head[0].decode() == sha, (head, sha)
    n += 1
    if b"\x00" in data[:8000]:
        continue
    text = data.decode("utf-8", "replace")
    path = paths[sha]
    for name, rx in RULES.items():
        for m in rx.finditer(text):
            s = m.group(0).replace("\n", " ")
            hits[name].add(f"{path} :: {s[:6]}…{s[-4:] if len(s) > 12 else ''}")
proc.wait()
print(f"{n} blobs scanned")
for name, hs in hits.items():
    print(f"== {name}: {len(hs)}")
    for h in sorted(hs)[:12]:
        print("   ", h)
