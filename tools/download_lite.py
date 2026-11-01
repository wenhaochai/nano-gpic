"""Download GPIC Lite train shards (0-800), val shards, eval prompts and reference stats.
Run on a vis node (needs internet). Resumable: hf_hub_download skips finished files."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]
REPO = "stanford-vision-lab/gpic"
files = [f"train/gpic_train_{i:05d}.tar" for i in range(801)]
files += [f"val/gpic_val_{i:05d}.tar" for i in range(32)]
files += ["gpic_test_50k.jsonl", "gpic_val_50k.jsonl", "README.md"]
files += [f"reference_stats/{w}_stats.npz" for w in ("test", "val", "lite")]
# small files first so eval can start early
files = files[-6:] + files[:-6]

def get(fn):
    for attempt in range(5):
        try:
            return hf_hub_download(REPO, fn, repo_type="dataset", local_dir=OUT)
        except Exception as e:
            print(f"retry {fn} {attempt}: {type(e).__name__} {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(fn)

t0 = time.time(); done = 0
with ThreadPoolExecutor(int(os.environ.get("NW", "8"))) as ex:
    futs = {ex.submit(get, f): f for f in files}
    for fu in as_completed(futs):
        done += 1
        try:
            fu.result()
        except Exception as e:
            print("FAILED", futs[fu], e, flush=True)
        if done % 10 == 0 or done == len(files):
            print(f"{done}/{len(files)} files {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
