#!/usr/bin/env python3
"""Parity of an attention lever on the whole denoiser (docs/mfu.md): the same weights and batch through the SDPA path
(production) and the lever's path, under bf16 autocast, against an fp32 reference (SDPA math backend, no autocast).

The zero-initialized layers (adaLN, final layer) get small random weights, so every block shapes the output. Batch: 8
images, a padded caption length T with prefix masks of different lengths. Reported per path: relative L2 error of the
output and of the gradient of every parameter against the fp32 reference (the worst parameter and the median). The lever
passes when its errors stay within 1.5x of the production path's errors (both are bf16 rounding).
usage: python test_attn_parity.py [--d 640] [--lever fa3|packed]
"""
import argparse, os, statistics, sys

import torch

sys.path.insert(0, os.environ.get("PIXGEN_SRC", os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=640)
    ap.add_argument("--nb", type=int, default=4)
    ap.add_argument("--nt", type=int, default=2)
    ap.add_argument("--lever", default="fa3")
    ap.add_argument("--T", type=int, default=56)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    import src.models.transformer.JiT_T2I as M
    if a.lever in ("fa3", "packed"):
        from src.models.layers.fa3_op import fa3_attention, fa3_varlen
        M.fa3_attention, M.fa3_varlen = fa3_attention, fa3_varlen
    dev = "cuda"
    torch.manual_seed(0)
    net = M.JiT_T2I(input_size=256, in_channels=3, num_groups=a.d // 64, hidden_size=a.d, num_blocks=a.nb,
                    num_text_blocks=a.nt, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256).to(dev)
    with torch.no_grad():
        for n, p in net.named_parameters():
            if p.abs().max() == 0:
                p.normal_(0, 0.02)
    B, T = 8, a.T
    g = torch.Generator(device=dev).manual_seed(1)
    x = torch.rand(B, 3, 256, 256, device=dev, generator=g) * 2 - 1
    t = torch.rand(B, device=dev, generator=g)
    lens = torch.tensor([T, T - 3, T - 7, T - 1, 5, T - 20, 1, T - 2], device=dev)
    mask = torch.arange(T, device=dev)[None] < lens[:, None]
    y = (torch.randn(B, T, 2048, device=dev, generator=g) * mask[..., None]).to(torch.bfloat16)
    dout = torch.randn(B, 3, 256, 256, device=dev, generator=g)

    if a.lever == "packed":   # lever E: the captions back to back, as the data path will deliver them
        sys.path.insert(0, os.path.dirname(__file__))
        from step_bench import make_pack, pack_rows
        pk = make_pack(lens.tolist(), T, dev)
        y_p = pack_rows(y[mask], pk)

    def run(path, autocast, fn=None):
        M._ATTN = "fa3" if path == "packed" else path
        net.zero_grad(set_to_none=True)
        f = fn or net
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            if path == "packed":
                out = f(x, t, y_p, y_pack=pk)
            elif path == "math":
                M._ATTN = "sdpa"
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    out = f(x, t, y.float() if not autocast else y, y_mask=mask)
            else:
                out = f(x, t, y, y_mask=mask)
        out.float().backward(dout)
        return out.detach().float(), {n: p.grad.detach().float().clone() for n, p in net.named_parameters() if p.grad is not None}

    ref_o, ref_g = run("math", False)
    res = {}
    for path in ("sdpa", a.lever):
        fn = torch.compile(net) if a.compile else None
        o, gr = run(path, True, fn)
        rel = {n: ((gr[n] - ref_g[n]).norm() / ref_g[n].norm().clamp_min(1e-30)).item() for n in ref_g}
        worst = max(rel, key=rel.get)
        res[path] = (((o - ref_o).norm() / ref_o.norm()).item(), rel[worst], worst, statistics.median(rel.values()))
        print(f"{path:6s} out rel err {res[path][0]:.3e}  grad rel err: worst {res[path][1]:.3e} ({worst}), median "
              f"{res[path][3]:.3e}", flush=True)
    s, l = res["sdpa"], res[a.lever]
    ok = l[0] <= 1.5 * s[0] and l[1] <= 1.5 * s[1] and l[3] <= 1.5 * s[3]
    print(f"PARITY {'PASS' if ok else 'FAIL'}: {a.lever} / sdpa error ratio: out {l[0] / s[0]:.2f}, worst grad "
          f"{l[1] / s[1]:.2f}, median grad {l[3] / s[3]:.2f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
