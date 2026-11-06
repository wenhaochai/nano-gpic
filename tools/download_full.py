"""Download the rest of the GPIC full train set (shards 801-7999; Lite is 0-800) and the full set's reference stats into data/gpic.
Run on a vis node (needs internet). Resumable: hf_hub_download skips finished files.
usage: download_full.py <out dir> [first shard] [last shard]"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]
first = int(sys.argv[2]) if len(sys.argv) > 2 else 801
last = int(sys.argv[3]) if len(sys.argv) > 3 else 7999
REPO = "stanford-vision-lab/gpic"
files = ["reference_stats/full_stats.npz"] + [f"train/gpic_train_{i:05d}.tar" for i in range(first, last + 1)]

def get(fn):
    for attempt in range(8):
        try:
            return hf_hub_download(REPO, fn, repo_type="dataset", local_dir=OUT)
        except Exception as e:
            print(f"retry {fn} {attempt}: {type(e).__name__} {e}", flush=True)
            time.sleep(min(600, 15 * 2 ** attempt))
    raise RuntimeError(fn)

t0 = time.time(); done = 0; failed = []
with ThreadPoolExecutor(int(os.environ.get("NW", "8"))) as ex:
    futs = {ex.submit(get, f): f for f in files}
    for fu in as_completed(futs):
        done += 1
        try:
            fu.result()
        except Exception as e:
            failed.append(futs[fu]); print("FAILED", futs[fu], e, flush=True)
        if done % 25 == 0 or done == len(files):
            print(f"{done}/{len(files)} files {time.time() - t0:.0f}s", flush=True)
print("DONE" if not failed else f"DONE with {len(failed)} failed: {failed[:20]}", flush=True)
