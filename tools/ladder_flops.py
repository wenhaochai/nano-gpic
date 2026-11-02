"""Ladder sizes, FLOPs per image and steps for an iso-FLOP budget, with the padded text length measured on real
captions under the option-B grouping (windows of 1024 in stream order, sorted, cut into per-GPU batches of 32, each
padded to its longest caption rounded up to a multiple of 8, at most 320).

python tools/ladder_flops.py [--shards 0 200 400 600 800] [--budget 1.426e19] [--out docs/ladder_runs.json]
FLOPs: the denoiser's training FLOPs (forward x 3) from its matmuls and attention, as in docs/ladder_plan.md
(1392 GFLOP/img for d1536 16+4 at T=300); the frozen text encoder is not counted.
"""
import argparse
import json
import math
import random
import tarfile

L, P, B = 256, 768, 256   # image tokens, patch dim, bottleneck
SIZES = [("d640", 640, 9, 2), ("d768", 768, 10, 2), ("d896", 896, 11, 2), ("d1024", 1024, 12, 3), ("d1280", 1280, 14, 3)]


def fwd(d, nb, nt, T, T2):
    """Forward FLOPs per image for text length T (E[T]) and E[T^2] (attention among text tokens)."""
    main = nb * (32 * L * d * d + 4 * T * d * d + 4 * L * (L + T) * d)
    text = nt * (32 * T * d * d + 4 * T2 * d)
    emb = 2 * L * P * B + 2 * L * B * d + 2 * T * 2048 * d + 2 * L * d * P
    return main + text + emb


def params(d, nb, nt):
    return nb * 24 * d * d + nt * 22 * d * d + P * B + B * d + 2048 * d + 320 * d + 256 * d + d * d + d * P


def check():
    """The formulas against the real module: parameter count (meta device) and forward FLOPs (FlopCounterMode, CUDA)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from src.models.transformer.JiT_T2I import JiT_T2I
    for name, d, nb, nt in SIZES + [("ref", 1536, 16, 4)]:
        kw = dict(input_size=256, in_channels=3, num_groups=d // 64, hidden_size=d, num_blocks=nb, num_text_blocks=nt,
                  patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256)
        with torch.device("meta"):
            n = sum(p.numel() for p in JiT_T2I(**kw).parameters() if p.requires_grad)
        msg = f"{name:6s} params real {n/1e6:7.2f}M formula {params(d, nb, nt)/1e6:7.2f}M ({params(d, nb, nt)/n-1:+.2%})"
        if name in ("d640", "d1280") and torch.cuda.is_available():
            m = JiT_T2I(**kw).cuda()
            for T in (64, 192):
                x = torch.randn(2, 3, 256, 256, device="cuda")
                y = torch.randn(2, T, 2048, device="cuda")
                mask = torch.ones(2, T, dtype=torch.bool, device="cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), FlopCounterMode(display=False) as fc:
                    m(x, torch.rand(2, device="cuda"), y, y_mask=mask)
                msg += f" | T={T} fwd GFLOP/img real {fc.get_total_flops()/2e9:.2f} formula {fwd(d, nb, nt, T, T*T)/1e9:.2f}"
            del m
            torch.cuda.empty_cache()
        print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 200, 400, 600, 800])
    ap.add_argument("--budget", type=float, default=40000 * 256 * 3 * fwd(1536, 16, 4, 300, 300 ** 2))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="docs/ladder_runs.json")
    ap.add_argument("--check", action="store_true", help="only compare the formulas with the real module")
    ap.add_argument("--pad-multiple", type=int, default=8)
    a = ap.parse_args()
    if a.check:
        return check()
    from transformers import Qwen2Tokenizer
    tok = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-1.7B", padding_side="right")
    lens = []
    for i in a.shards:
        with tarfile.open(f"/scratch/gpfs/GROUP/USER/data/gpic/train/gpic_train_{i:05d}.tar") as t:
            caps = [json.loads(t.extractfile(m).read())["caption"] for m in t if m.name.endswith(".json")]
        lens += [min(len(x), 320) for x in tok(caps, truncation=True, max_length=320)["input_ids"]]
    random.seed(0)
    random.shuffle(lens)   # stand-in for the shard shuffle + sample shuffle buffer
    Ts = []
    for w in range(0, len(lens) - 1023, 1024):
        win = sorted(lens[w:w + 1024])
        for c in range(0, 1024, 32):
            Ts.append(min(-(-max(win[c:c + 32]) // a.pad_multiple) * a.pad_multiple, 320))
    ET = sum(Ts) / len(Ts)
    ET2 = sum(t * t for t in Ts) / len(Ts)
    hist = {t: round(Ts.count(t) / len(Ts), 4) for t in sorted(set(Ts))}
    print(f"captions {len(lens)}, batches {len(Ts)}: mean caption {sum(lens)/len(lens):.1f} tokens, "
          f"mean padded T {ET:.1f}, histogram {hist}")
    print(f"budget C0 = {a.budget:.4e} FLOPs (reference run: 40k steps x 256 at T=300)")
    runs = []
    for name, d, nb, nt in SIZES:
        g = 3 * fwd(d, nb, nt, ET, ET2)
        N = params(d, nb, nt)
        steps = a.budget / (a.batch * g)
        steps = int(round(steps / 4000) * 4000)
        imgs = steps * a.batch
        r = {"size": name, "d": d, "nb": nb, "nt": nt, "heads": d // 64, "params_M": round(N / 1e6, 1),
             "gflop_per_img": round(g / 1e9, 1), "max_steps": steps, "ms_every": steps // 4,
             "tpp": round(imgs * L / N, 1), "lite_epochs": round(imgs / 10.1e6, 2),
             "budget_flops": imgs * g}
        runs.append(r)
        print(f"{name:6s} {nb}+{nt} {N/1e6:6.1f}M  {g/1e9:6.1f} GFLOP/img  {steps:7d} steps (ms every {steps//4})  "
              f"TPP {r['tpp']:6.1f}  {r['lite_epochs']:5.2f} epochs  C {imgs*g:.3e}")
    json.dump({"budget": a.budget, "mean_padded_T": ET, "T_hist": hist, "runs": runs}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
