"""Fetch the 50K GPIC test images behind the official reference statistics (data/gpic/reference_stats/test_stats.npz), for
reference statistics at other resolutions (resolution sweep, tmp/ladder_prof/res/PLAN.md section 5, item 4).

test_stats.npz names its images: `key` (50K image keys) and `tar_idx` (the test tar of each, gpic_test_<tar_idx>.tar; 346-441
images in each of the 128 tars). This script downloads one test tar at a time from the Hugging Face dataset, copies the reference
images (the original bytes, .jpg or .png) into <out>/ref_<tar>.tar, and deletes the downloaded tar, so at most <workers> test tars
(about 1 GB each) sit on disk at once. A tar whose ref_<tar>.tar already holds all its images is skipped, so a rerun does the rest.
Writes <out>/manifest.json (images per tar) and exits 1 unless all 50,000 images were found.
usage (vis node: needs the internet and the HF login in $HF_HOME): fetch_test_reference_images.py <out dir> [--workers 3]
"""
import argparse, io, json, os, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

WC = "/scratch/gpfs/GROUP/USER"
REPO = "stanford-vision-lab/giant-permissive-image-corpus"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def done_keys(path):
    if not os.path.exists(path):
        return set()
    try:
        with tarfile.open(path) as t:
            return {os.path.splitext(m.name)[0] for m in t if m.isfile()}
    except (tarfile.TarError, OSError, EOFError):
        return set()   # a partial tar from a killed run: fetched again


def download(tar, dl_dir, tries=5):
    from huggingface_hub import hf_hub_download
    for i in range(tries):   # the Xet CDN drops a request now and then (2026-11-17: one CAS client error in 57 tars)
        try:
            return hf_hub_download(REPO, f"test/gpic_test_{tar:05d}.tar", repo_type="dataset", local_dir=dl_dir)
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"tar {tar:05d}: download attempt {i + 1} failed ({type(e).__name__}); retry in {30 * (i + 1)} s", flush=True)
            time.sleep(30 * (i + 1))


def one_tar(tar, want, out, dl_dir):
    dst, miss_f = f"{out}/ref_{tar:05d}.tar", f"{out}/missing_{tar:05d}.json"
    recorded = set(json.load(open(miss_f))["missing"]) if os.path.exists(miss_f) else set()
    have = done_keys(dst)
    if have | recorded >= want:
        return tar, len(have & want), f"done before, missing {len(want - have)}"
    t0 = time.time()
    src = download(tar, dl_dir)
    t1 = time.time()
    found = set()
    tmp = dst + ".partial"
    with tarfile.open(src) as tin, tarfile.open(tmp, "w") as tout:
        for m in tin:
            stem, ext = os.path.splitext(os.path.basename(m.name))
            if not m.isfile() or ext.lower() not in IMG_EXT or stem not in want or stem in found:
                continue
            data = tin.extractfile(m).read()
            info = tarfile.TarInfo(stem + ext.lower())
            info.size = len(data)
            tout.addfile(info, io.BytesIO(data))
            found.add(stem)
        miss = sorted(want - found)
        if miss:   # what the tar holds under those keys, if anything (before the tar is deleted)
            members = [[m.name, m.size] for m in tin.getmembers() if any(k in m.name for k in miss)]
            json.dump({"missing": miss, "members": members}, open(miss_f, "w"))
    os.replace(tmp, dst)
    os.remove(src)
    return tar, len(found), f"download {t1 - t0:.0f} s, copy {time.time() - t1:.0f} s, missing {len(miss)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref = np.load(f"{WC}/data/gpic/reference_stats/test_stats.npz")
    keys, tars = ref["key"].tolist(), ref["tar_idx"].tolist()
    want = {}
    for k, t in zip(keys, tars):
        want.setdefault(int(t), set()).add(k)
    dl_dir = f"{WC}/data/gpic"   # hf_hub_download puts test/gpic_test_<tar>.tar under it
    counts = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for tar, n, note in ex.map(lambda t: one_tar(t, want[t], a.out, dl_dir), sorted(want)):
            counts[tar] = n
            print(f"tar {tar:05d}: {n}/{len(want[tar])} images ({note})", flush=True)
    total = sum(counts.values())
    json.dump({"images": total, "per_tar": {f"{t:05d}": counts[t] for t in sorted(counts)}}, open(f"{a.out}/manifest.json", "w"), indent=0)
    print(f"total {total} of {len(keys)}")
    sys.exit(0 if total == len(keys) else 1)


if __name__ == "__main__":
    main()
