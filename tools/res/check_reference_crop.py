"""Which crop made the official reference? DINOv2 features (CPU, fp32) of the first <n> reference images of one fetched test tar,
centre-cropped to 256 two ways, against the features stored in test_stats.npz for the same keys:
  toolkit  gpic_eval.utils.resize_center_crop: crop the centre square, then PIL bicubic to 256 (the toolkit's image loader);
  adm      src.data.dataset.augmentation.center_crop_arr: short side to 256 (box halving, then bicubic), then the centre 256 (the
           baseline's val_transform, which the sweep's held-out tool uses).
Prints per crop the median and worst cosine similarity and relative L2 distance to the stored features.
usage (vis node, CUDA_VISIBLE_DEVICES= so torch stays off the shared GPU): check_reference_crop.py <ref_<tar>.tar> [n]
"""
import io, os, sys, tarfile

import numpy as np
import torch
from PIL import Image

WC = "/scratch/gpfs/GROUP/USER"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../baselines/PixelGen"))
from src.data.dataset.augmentation import center_crop_arr   # noqa: E402
from gpic_eval.utils import resize_center_crop               # noqa: E402
from gpic_eval.backends.dino import build_dinov2, _dino_transform   # noqa: E402

path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100
ref = np.load(f"{WC}/data/gpic/reference_stats/test_stats.npz")
row = {k: i for i, k in enumerate(ref["key"].tolist())}
emb = ref["dino_embeddings"]
imgs, rows = [], []
with tarfile.open(path) as t:
    for m in t:
        if len(imgs) == n:
            break
        stem = os.path.splitext(m.name)[0]
        imgs.append(Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB"))
        rows.append(row[stem])
want = emb[rows].astype(np.float64)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
model = build_dinov2(torch.device("cpu"))
for name, crop in [("toolkit", lambda im: resize_center_crop(im, 256)), ("adm", lambda im: center_crop_arr(im, 256))]:
    x = torch.stack([_dino_transform(np.asarray(crop(im))) for im in imgs])
    with torch.no_grad():
        f = torch.cat([model(x[i:i + 16]) for i in range(0, len(x), 16)]).double().numpy()
    cos = (f * want).sum(1) / np.linalg.norm(f, axis=1) / np.linalg.norm(want, axis=1)
    rel = np.linalg.norm(f - want, axis=1) / np.linalg.norm(want, axis=1)
    print(f"{name:8s} n={len(f)}  cosine median {np.median(cos):.6f} min {cos.min():.6f}  rel L2 median {np.median(rel):.2e} max {rel.max():.2e}", flush=True)
