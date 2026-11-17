"""Crop the fetched reference test images (fetch_test_reference_images.py) to R x R for reference statistics at R.

The crop is the toolkit's own, gpic_eval.utils.resize_center_crop (the centre square, then PIL bicubic to R): at R = 256 it gives the
DINOv2 features stored in test_stats.npz to a relative L2 distance of 4e-6 (check_reference_crop.py, 64 images of tar 0), while the
baseline's center_crop_arr is 7% off. One decode per image serves every R. Writes <out>/test_r<R>.npy (NHWC uint8, rows in the order
of test_stats.npz's key) and <out>/rows.npy (the row indices written: a key that no test tar holds stays out of the statistics).
usage (vis node or a pilot's CPUs): crop_reference_images.py <ref tar dir> <out dir> <R> [<R> ...] [--procs 6]
"""
import argparse, glob, io, os, sys, tarfile, time
from multiprocessing import Pool

import numpy as np
from PIL import Image

WC = "/scratch/gpfs/GROUP/USER"
G = {}


def init(rows_of, outs):
    G["rows"] = rows_of
    G["arrs"] = {r: np.load(p, mmap_mode="r+") for r, p in outs.items()}


def work(path):
    from gpic_eval.utils import resize_center_crop
    done = []
    with tarfile.open(path) as t:
        for m in t:
            i = G["rows"][os.path.splitext(m.name)[0]]
            img = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
            for r, a in G["arrs"].items():
                a[i] = resize_center_crop(img, r)
            done.append(i)
    for a in G["arrs"].values():
        a.flush()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tars"); ap.add_argument("out"); ap.add_argument("res", type=int, nargs="+")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    keys = np.load(f"{WC}/data/gpic/reference_stats/test_stats.npz")["key"].tolist()
    rows_of = {k: i for i, k in enumerate(keys)}
    outs = {}
    for r in a.res:
        outs[r] = f"{a.out}/test_r{r}.npy"
        np.lib.format.open_memmap(outs[r], mode="w+", dtype=np.uint8, shape=(len(keys), r, r, 3)).flush()
    paths = sorted(glob.glob(f"{a.tars}/ref_*.tar"))
    t0, rows = time.time(), []
    with Pool(a.procs, initializer=init, initargs=(rows_of, outs)) as p:
        for j, done in enumerate(p.imap_unordered(work, paths)):
            rows += done
            if j % 16 == 15:
                print(f"{j + 1}/{len(paths)} tars, {len(rows)} images, {time.time() - t0:.0f} s", flush=True)
    rows = np.array(sorted(rows), dtype=np.int64)
    assert len(rows) == len(set(rows.tolist())), "an image was written twice"
    np.save(f"{a.out}/rows.npy", rows)
    print(f"{len(rows)} of {len(keys)} images at {a.res} in {time.time() - t0:.0f} s; missing rows {sorted(set(range(len(keys))) - set(rows.tolist()))}")


if __name__ == "__main__":
    main()
