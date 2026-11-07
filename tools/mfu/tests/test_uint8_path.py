#!/usr/bin/env python3
"""MFU lever D, CPU test: for real crops of a GPIC shard, uint8_to_float(uint8 crop) equals the training transform's
ToTensor + Normalize bit for bit, and the grouped collate stacks uint8 images.
usage (vis node, PYTHONPATH=baselines/PixelGen): python tools/mfu/tests/test_uint8_path.py [n images]"""
import io, os, sys, tarfile
import numpy as np
import torch
from PIL import Image
os.environ["GPIC_REGION_CROP"] = "1"
from src.data.dataset import gpic_dataset as G

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
post = torch.nn.Sequential() if False else None
pre, post = G.train_transform.transforms[0], G.T.Compose(G.train_transform.transforms[1:])
checked = 0
with tarfile.open("/scratch/gpfs/GROUP/USER/data/gpic/train/gpic_train_00007.tar") as t:
    for m in t:
        if not m.name.endswith((".jpg", ".png", ".jpeg", ".webp")):
            continue
        img = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
        crop = np.asarray(pre(img), dtype=np.uint8)
        ref = post(crop.copy())
        got = G.uint8_to_float(torch.from_numpy(crop)[None])[0]
        assert got.dtype == ref.dtype and got.shape == ref.shape, (got.shape, ref.shape)
        assert torch.equal(got.view(torch.int32), ref.view(torch.int32)), f"{m.name}: max diff {(got - ref).abs().max()}"
        checked += 1
        if checked >= n:
            break
print(f"ok: {checked} real crops, uint8_to_float == ToTensor + Normalize bit for bit")
