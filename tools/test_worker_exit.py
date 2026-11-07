"""A process with a live pinned DataLoader iterator takes 5 s per persistent worker to exit, because torch's exit hook waits for
workers that were never told to stop. GPICWebDataset.stop_workers (the end of baselines/PixelGen/main.py) makes the exit immediate.
On a vis node with a GPU, real shards and the real text cache:
  PYTHONPATH=baselines/PixelGen python tools/test_worker_exit.py <tar_dir> <emb_dir>
"""
import subprocess
import sys
import time

WORKERS = 8


def child(tar_dir, emb_dir, stop):
    global it   # alive until the interpreter exits, as main.py's cli keeps the trainer and its iterator
    from src.data.dataset.gpic_dataset import GPICWebDataset
    ds = GPICWebDataset(tar_dir=tar_dir, tar_pattern=f"gpic_train_{{00000..{WORKERS - 1:05d}}}.tar", group_text_len=True, group_window=1024,
                        group_batch=32, text_max_length=320, pad_multiple=8, emb_dir=emb_dir)
    it = iter(ds.build_dataloader(32, WORKERS))
    for _ in range(WORKERS * 2):
        next(it)
    if stop:
        ds.stop_workers()
    print("end of main", time.time(), flush=True)


def main():
    tar_dir, emb_dir = sys.argv[1:3]
    if len(sys.argv) > 3:
        child(tar_dir, emb_dir, sys.argv[3] == "1")
        return
    ok = True
    for arg, lo, hi in (("0", WORKERS * 4, WORKERS * 6), ("1", 0, 5)):
        p = subprocess.run([sys.executable, __file__, tar_dir, emb_dir, arg], capture_output=True, text=True, timeout=400)
        end = float(p.stdout.strip().split()[-1])
        took = time.time() - end
        good = p.returncode == 0 and lo <= took <= hi
        ok &= good
        print(f"stop_workers={arg}: exit {took:.1f} s after the end of main, rc {p.returncode}: {'ok' if good else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
