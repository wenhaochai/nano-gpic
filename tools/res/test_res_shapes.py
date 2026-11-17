"""CPU test of the resolution sweep's code (genmodel 2026-11-17, tmp/ladder_prof/res/PLAN.md).
1. The crops give R x R at R = 64, 128, 256, 512 for GPIC-like sizes (long side 640), and GPIC_RES sets the module's crop size.
2. JiT_T2I builds and runs at every (R, patch) of the plan, through F.unfold / F.fold and through lever P (patchify), with
   output shape = input shape and the two paths equal; the position embedding has (R / patch)^2 rows.
usage (vis node, CUDA_VISIBLE_DEVICES=): python -I tools/res/test_res_shapes.py"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PG = os.path.join(ROOT, "baselines", "PixelGen")
sys.path.insert(0, PG)
import numpy as np
import torch
from PIL import Image
from src.data.dataset.augmentation import center_crop_arr, random_crop_arr, random_crop_arr_region

rng = np.random.default_rng(0)
for w, h in ((640, 360), (640, 480), (500, 333), (333, 500), (640, 640), (750, 512)):
    img = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    for R in (64, 128, 256, 512):
        for f in (center_crop_arr, random_crop_arr, random_crop_arr_region):
            a = np.asarray(f(img, R))
            assert a.shape == (R, R, 3), (f.__name__, w, h, R, a.shape)
print("1a. crops: R x R for 6 sizes x 4 resolutions x 3 crops")

for R in (64, 128, 256, 512):
    code = ("import sys; sys.path.insert(0, %r); from src.data.dataset import gpic_dataset as g; from PIL import Image; import numpy as np; "
            "im = Image.fromarray(np.zeros((360, 640, 3), np.uint8)); "
            "assert g.RES == %d, g.RES; assert tuple(g.train_transform(im).shape) == (3, %d, %d); assert tuple(g.val_transform(im).shape) == (3, %d, %d)"
            % (PG, R, R, R, R, R))
    env = dict(os.environ, GPIC_RES=str(R))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, (R, r.stderr[-800:])
r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); from src.data.dataset import gpic_dataset" % PG],
                   env=dict(os.environ, GPIC_RES="300"), capture_output=True, text=True)
assert r.returncode != 0 and "GPIC_RES=300" in r.stderr, "an unplanned resolution must be refused"
print("1b. GPIC_RES: train and val transforms give 3 x R x R at 64, 128, 256, 512; 300 is refused")

from src.models.transformer import JiT_T2I as M
for R, p in ((64, 16), (128, 16), (256, 16), (512, 16), (64, 4), (128, 8), (512, 32)):
    torch.manual_seed(0)
    m = M.JiT_T2I(input_size=R, in_channels=3, num_groups=2, hidden_size=64, num_blocks=2, num_text_blocks=1, patch_size=p,
                  txt_embed_dim=32, txt_max_length=8, bottleneck_dim=16).eval()
    with torch.no_grad():
        for prm in m.parameters():
            prm.add_(0.02 * torch.randn_like(prm))   # the final layer starts at zero: perturb so the output carries the whole path
    assert m.pos_embed.shape[1] == (R // p) ** 2, (R, p, m.pos_embed.shape)
    assert int(round(m.num_patches ** 0.5)) * m.patch_size == R, (R, p)   # compile_warmup.py's dummy image side
    x, t, y = torch.randn(2, 3, R, R), torch.rand(2), torch.randn(2, 8, 32)
    with torch.no_grad():
        M._PATCHIFY = False
        a = m(x, t, y)
        M._PATCHIFY = True
        b = m(x, t, y)
        M._PATCHIFY = False
    assert a.shape == x.shape, (R, p, a.shape)
    assert torch.equal(a, b), (R, p, float((a - b).abs().max()))
    print(f"2. JiT_T2I R={R} patch={p}: {(R // p) ** 2} tokens, output {tuple(a.shape)}, unfold == patchify")
print("ok")
