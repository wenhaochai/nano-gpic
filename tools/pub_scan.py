"""Secret/privacy scan of every blob reachable in a repo (rules from tmp/ssov_scan/pub_scan.py). Masked output."""
import re, subprocess, sys, collections
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
    "cluster_ids": re.compile(r"USER|GROUP|group|USER"),
}
objs = subprocess.run(["git", "-C", repo, "rev-list", "--objects", sys.argv[2] if len(sys.argv) > 2 else "--all"], capture_output=True, text=True).stdout.splitlines()
blobs = {}
for line in objs:
    parts = line.split(" ", 1)
    if len(parts) == 2:
        blobs.setdefault(parts[0], parts[1])
hits = collections.defaultdict(set)
n = 0
for sha, path in blobs.items():
    t = subprocess.run(["git", "-C", repo, "cat-file", "-t", sha], capture_output=True, text=True).stdout.strip()
    if t != "blob":
        continue
    n += 1
    data = subprocess.run(["git", "-C", repo, "cat-file", "blob", sha], capture_output=True).stdout
    if b"\x00" in data[:8000]:
        continue
    text = data.decode("utf-8", "replace")
    for name, rx in RULES.items():
        for m in rx.finditer(text):
            s = m.group(0).replace("\n", " ")
            hits[name].add(f"{path} :: {s[:6]}…{s[-4:] if len(s) > 12 else ''}")
print(f"{n} blobs scanned")
for name, hs in hits.items():
    print(f"== {name}: {len(hs)}")
    for h in sorted(hs)[:12]:
        print("   ", h)
