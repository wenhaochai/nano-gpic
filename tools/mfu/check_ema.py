#!/usr/bin/env python3
"""EMA check across runs of the same config (MFU lever B, FSDP2), without the initial weights: for the tensors whose
initial value the code fixes (zero: every Linear bias, the adaLN modulation layers, the final linear layer; these are
sharded under FSDP2; one: the RMSNorm weights, replicated under FSDP2), the EMA's share of the denoiser's travel
f = |ema - init| / |den - init| and cos(ema - init, den - init), per group and the range over tensors. With the same
decay and the same number of steps, a correct EMA gives about the same f and cos in every run; an EMA that was not
updated gives f = 0, one that lerps the wrong shard or in the wrong dtype a different f. pos_embed (fixed): ema == den.
usage: python check_ema.py <ckpt>...
"""
import re, sys
import torch

ZERO = re.compile(r"(adaLN_modulation\.\d+\.(weight|bias)|final_layer\.linear\.(weight|bias)|proj2\.bias|mlp\.\d+\.bias"
                  r"|y_embedder\.proj\.bias|attn\.proj\.bias)$")
ONES = re.compile(r"(norm1|norm2|q_norm|k_norm|y_embedder\.norm)\.weight$")


def main():
    for p in sys.argv[1:]:
        c = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        sd = c["state_dict"]
        out = [f"step {c['global_step']} {p.split('gpic-speed/')[-1]}"]
        for name, pat, init in (("zero-init (sharded)", ZERO, 0.0), ("one-init (replicated)", ONES, 1.0)):
            ee = dd = ed = 0.0
            fs = []
            n = 0
            for k, den in sd.items():
                if not k.startswith("denoiser.") or not pat.search(k):
                    continue
                ema = sd["ema_" + k].double()
                a, b = ema - init, den.double() - init
                ee += float(a.pow(2).sum()); dd += float(b.pow(2).sum()); ed += float((a * b).sum())
                fs.append(float(a.norm() / b.norm().clamp_min(1e-30)))
                n += 1
            out.append(f"{name}: {n} tensors, f {ee ** .5 / dd ** .5:.4f} (tensors {min(fs):.4f}..{max(fs):.4f}), "
                       f"cos {ed / (ee * dd) ** .5:.4f}")
        pe = sd["denoiser.pos_embed"], sd["ema_denoiser.pos_embed"]
        out.append(f"pos_embed ema == den: {bool(torch.equal(*pe))}")
        print("\n  ".join(out))


if __name__ == "__main__":
    main()
