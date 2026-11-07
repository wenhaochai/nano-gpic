"""Stack the sample_check PNGs of one checkpoint into one labelled sheet. python tools/sample_grid.py <dir> <out.png> <title>"""
import sys
from PIL import Image, ImageDraw, ImageFont
d, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [("ema_denoiser", "EMA"), ("denoiser", "raw")]
items = [(w, lab, cfg, n) for w, lab in rows for cfg in ("1", "4") for n in ("shared", "each")]
W, H, L, T = 1024, 256, 260, 40
sheet = Image.new("RGB", (L + W, T + H * len(items)), "white")
dr = ImageDraw.Draw(sheet)
try:
    f = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 20)
except Exception:
    f = ImageFont.load_default()
dr.text((10, 8), title, fill="black", font=f)
for i, (w, lab, cfg, n) in enumerate(items):
    im = Image.open(f"{d}/{w}_cfg{cfg}_{n}.png").convert("RGB")
    sheet.paste(im, (L, T + i * H))
    dr.text((10, T + i * H + 90), f"{lab} weights\nCFG {cfg}\n{'shared noise (official)' if n == 'shared' else 'noise per prompt'}", fill="black", font=f)
sheet.save(out)
print(out, sheet.size)
