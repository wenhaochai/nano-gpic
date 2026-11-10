#!/usr/bin/env python3
"""CPU test of rewrite_twins.py: synthetic checkpoints for every branch, then one real old-layout file (a copy).
usage: test_rewrite_twins.py <work dir> <real old-layout ckpt>"""
import collections, hashlib, os, shutil, subprocess, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
work, real = sys.argv[1], sys.argv[2]
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work + "/r/a")
fails = []


def check(name, ok, text=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + text if text else ''}", flush=True)
    if not ok:
        fails.append(name)


def raw(t):
    return bytes(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy()) if t.numel() else b""


def md5(f):
    h = hashlib.md5()
    with open(f, "rb") as g:
        for b in iter(lambda: g.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def make(path, hidden=False):
    g = torch.Generator().manual_seed(1)
    sd = collections.OrderedDict()
    sd["denoiser.a"] = torch.randn(300, 400, generator=g)
    sd["denoiser.b"] = torch.randn(70000, generator=g).to(torch.bfloat16)
    sd["denoiser.c"] = torch.tensor(7, dtype=torch.int64)
    d = torch.randn(40000, generator=g)
    d[5], d[6] = float("nan"), -0.0
    sd["denoiser.d"] = d
    sd["denoiser.f"] = torch.randn(5000, generator=g)
    sd["other.x"] = torch.randn(1000, generator=g)
    for k in "abcd":
        sd["ema_denoiser." + k] = sd["denoiser." + k].clone()
    sd["ema_denoiser.e"] = torch.randn(3000, generator=g)                 # no raw tensor of this name
    sd["ema_denoiser.f"] = sd["denoiser.f"].double()                      # another dtype: not a pair
    if hidden:   # one element outside the 3 sampled slices of 16,384
        sd["ema_denoiser.b"].view(torch.int16)[20000] ^= 1
    sd._metadata = collections.OrderedDict({"": {"version": 1}, "denoiser": {"version": 2}})
    ck = {"epoch": 3, "global_step": 96, "pytorch-lightning_version": "2.6.6",
          "gpic_shard_clock": {"rank0": [1, 2, 3], "seed": 20261106}, "state_dict": sd}
    torch.save(ck, path)


def scan_line(rel, size=None, mtime=None):
    st = os.lstat(f"{work}/r/{rel}")
    return f"./{rel}\t{st.st_size if size is None else size}\t{(st.st_mtime_ns / 1e9 if mtime is None else mtime):.10f}\tequal\tepoch\n"


def run(scan_text, log, *flags):
    open(work + "/scan.tsv", "w").write(scan_text)
    p = subprocess.run([sys.executable, os.path.join(HERE, "..", "rewrite_twins.py"), *flags, log, work + "/scan.tsv"],
                       env=dict(os.environ, TWINS_ROOT=work + "/r"), capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def status(log, rel):
    for l in open(log):
        c = l.rstrip("\n").split("\t")
        if c[0] == "./" + rel:
            s = c[1]
    return s


# 1 the plain case, with a leftover of a killed run beside it
f = work + "/r/a/ms-step=000096.ckpt"
make(f)
os.chmod(f, 0o640)
os.utime(f, ns=(1794000000_000000000, 1794000000_123456789))
open(f + ".twinsrw_tmp", "w").write("part of a killed run")
before, m0 = os.lstat(f), md5(f)
old = torch.load(f, map_location="cpu", weights_only=False)
line = scan_line("a/ms-step=000096.ckpt")
rc, out = run(line, work + "/dry.tsv")
check("dry run changes nothing", rc == 0 and "would-rewrite" in out and md5(f) == m0 and not os.path.exists(work + "/dry.tsv")
      and os.path.exists(f + ".twinsrw_tmp"), out.strip().split("\n")[-1])
rc, out = run(line, work + "/log.tsv", "--go")
after = os.lstat(f)
new = torch.load(f, map_location="cpu", weights_only=False)
check("plain: status ok", rc == 0 and status(work + "/log.tsv", "a/ms-step=000096.ckpt") == "ok", out.strip().split("\n")[-1])
check("plain: smaller, mtime and mode kept, new inode, no temporary file",
      after.st_size < before.st_size and after.st_mtime_ns == before.st_mtime_ns and (after.st_mode & 0o777) == 0o640
      and after.st_ino != before.st_ino and not os.path.exists(f + ".twinsrw_tmp"), f"{before.st_size} -> {after.st_size} bytes")
a, b = old["state_dict"], new["state_dict"]
check("plain: keys, order, types, _metadata, other values",
      list(old) == list(new) and list(a) == list(b) and type(a) is type(b) and a._metadata == b._metadata
      and all(old[k] == new[k] for k in old if k != "state_dict"))
check("plain: every tensor bit for bit (NaN, -0.0, bf16, 0-dim)",
      all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and raw(a[k]) == raw(b[k]) for k in a))
check("plain: 4 pairs share storage, the others do not",
      all(b["ema_denoiser." + k].data_ptr() == b["denoiser." + k].data_ptr() for k in "abcd")
      and b["ema_denoiser.f"].data_ptr() != b["denoiser.f"].data_ptr())
w = torch.load(f, map_location="cpu", weights_only=True)
check("plain: loads with weights_only", all(raw(w["state_dict"][k]) == raw(a[k]) for k in a))
rc, out = run(line, work + "/log.tsv", "--go")
check("second run: already ok in the log, nothing done", rc == 0 and "0 to do, 1 already ok" in out and os.lstat(f).st_ino == after.st_ino, out.strip().split("\n")[0])
rc, out = run(scan_line("a/ms-step=000096.ckpt"), work + "/log2.tsv", "--go")
check("a twins-layout file: no pairs, not written again", rc == 0 and status(work + "/log2.tsv", "a/ms-step=000096.ckpt") == "skip-no-pairs"
      and os.lstat(f).st_ino == after.st_ino)

# 2 a difference outside the samples: found by the full comparison, the old file stays
h = work + "/r/a/ms-step=000192.ckpt"
make(h, hidden=True)
st, m0 = os.lstat(h), md5(h)
rc, out = run(scan_line("a/ms-step=000192.ckpt"), work + "/log3.tsv", "--go")
s = status(work + "/log3.tsv", "a/ms-step=000192.ckpt")
now = os.lstat(h)
check("hidden difference: FAIL-compare, exit 1, the old file untouched, no temporary file",
      rc == 1 and s.startswith("FAIL-compare") and "ema_denoiser.b" in s and (now.st_ino, now.st_mtime_ns) == (st.st_ino, st.st_mtime_ns)
      and md5(h) == m0 and not os.path.exists(h + ".twinsrw_tmp"), s)

# 3 files that are not as the scan listed them
for name, prep, want in (
        ("size", lambda rel: scan_line(rel, size=5), "skip-size-changed"),
        ("mtime", lambda rel: scan_line(rel, mtime=1.0), "skip-mtime-changed"),
        ("hardlink", lambda rel: (os.link(f"{work}/r/{rel}", f"{work}/r/{rel}.hard"), scan_line(rel))[1], "skip-hard-links"),
        ("symlink", lambda rel: (os.rename(f"{work}/r/{rel}", f"{work}/r/{rel}.real"), os.symlink(f"{work}/r/{rel}.real", f"{work}/r/{rel}"), scan_line(rel))[2], "skip-not-regular"),
        ("gone", lambda rel: (scan_line(rel), os.remove(f"{work}/r/{rel}"))[0], "skip-gone")):
    rel = f"a/ms-step=9{name}.ckpt"
    make(f"{work}/r/{rel}")
    line = prep(rel)
    m0 = md5(f"{work}/r/{rel}") if os.path.exists(f"{work}/r/{rel}") else None
    rc, out = run(line, f"{work}/log_{name}.tsv", "--go")
    s = status(f"{work}/log_{name}.tsv", rel)
    check(f"not as listed ({name}): {want}, exit 0, unchanged", rc == 0 and s.split(":")[0] == want
          and (m0 is None or md5(f"{work}/r/{rel}") == m0), s)

# 4 a real old-layout file (a copy with its mtime), two parts
os.makedirs(work + "/r/real")
for i in (1, 2):
    shutil.copy2(real, f"{work}/r/real/ms-step=00000{i}.ckpt")
lines = "".join(scan_line(f"real/ms-step=00000{i}.ckpt") for i in (1, 2))
for part in ("0/2", "1/2"):
    rc, out = run(lines, f"{work}/log_real{part[0]}.tsv", "--go", "--part", part)
    check(f"real file, part {part}", rc == 0 and "'ok': 1" in out, out.strip().split("\n")[-1])
src = torch.load(real, map_location="cpu", mmap=True, weights_only=False)
for i in (1, 2):
    g = f"{work}/r/real/ms-step=00000{i}.ckpt"
    new = torch.load(g, map_location="cpu", weights_only=False)   # as a reader without mmap
    a, b = src["state_dict"], new["state_dict"]
    n = sum(1 for k in b if k.startswith("ema_denoiser.") and b[k].data_ptr() == b[k[4:]].data_ptr())
    check(f"real file {i}: every key and tensor bit for bit, mtime kept",
          list(src) == list(new) and list(a) == list(b) and type(a) is type(b) and getattr(a, "_metadata", None) == getattr(b, "_metadata", None)
          and all(src[k] == new[k] for k in src if k != "state_dict")
          and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and raw(a[k]) == raw(b[k]) for k in a)
          and os.lstat(g).st_mtime_ns == os.lstat(real).st_mtime_ns,
          f"{os.path.getsize(real)} -> {os.path.getsize(g)} bytes (ratio {os.path.getsize(g) / os.path.getsize(real):.4f}), {len(a)} tensors, {n} shared pairs")
shutil.rmtree(work, ignore_errors=True)
print("TEST_REWRITE_TWINS " + ("PASS" if not fails else "FAIL " + "; ".join(fails)))
sys.exit(1 if fails else 0)
