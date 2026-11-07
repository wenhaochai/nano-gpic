"""perf C7 check: the device-side RoPE cache gives the same denoiser output as the CPU cache, under torch.compile,
with no recompilation and no pageable host-to-device copy per forward.

PYTHONPATH=baselines/PixelGen python tools/test_rope_cache.py [--blocks 2 --text-blocks 1 --batch 2]
Builds JiT_T2I with the job's dimensions (fewer blocks), compiles it like the LightningModel does (module.compile()),
runs 3 forwards with the new fetch_pos and 3 with the C4 version (tables cached on the CPU, .to(device) per call),
and compares outputs bit for bit. Counts Dynamo recompiles and pageable HtoD memcpys (torch.profiler) per version.
"""
import argparse
import types

import torch


def c4_fetch_pos(self, height, width, device):
    from src.models.layers.rope import precompute_freqs_cis_2d
    if (height, width) not in self.precompute_pos:
        pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width)
        self.precompute_pos[(height, width)] = (pos.real.contiguous().float(), pos.imag.contiguous().float())
    cos, sin = self.precompute_pos[(height, width)]
    return cos.to(device), sin.to(device)


def run(model, x, t, y, n=3):
    from torch._dynamo.utils import counters
    counters.clear()
    outs = []
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        for _ in range(n):
            outs.append(model(x, t, y).float())
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            outs.append(model(x, t, y).float())
            torch.cuda.synchronize()
    pageable = sum(1 for e in prof.events() if "Memcpy HtoD" in e.name and "Pageable" in e.name)
    recompiles = sum(v for k, v in counters["stats"].items() if k == "unique_graphs")
    return outs, pageable, recompiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--text-blocks", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    from src.models.transformer.JiT_T2I import JiT_T2I
    torch.manual_seed(0)
    dev = torch.device("cuda")
    kw = dict(input_size=256, in_channels=3, num_groups=24, hidden_size=1536, num_blocks=a.blocks,
              num_text_blocks=a.text_blocks, patch_size=16, txt_embed_dim=2048, txt_max_length=300,
              bottleneck_dim=256, weight_path=None, load_ema=False)
    x = torch.randn(a.batch, 3, 256, 256, device=dev)
    t = torch.rand(a.batch, device=dev)
    y = torch.randn(a.batch, 300, 2048, device=dev)
    res = {}
    for name in ("C7", "C4"):
        torch._dynamo.reset()
        torch.manual_seed(1)
        m = JiT_T2I(**kw).to(dev)
        for p in m.parameters():            # non-zero output layer so the output depends on every block
            p.data.add_(torch.randn_like(p) * 1e-3)
        if name == "C4":
            m.fetch_pos = types.MethodType(c4_fetch_pos, m)
        m.compile()
        res[name] = run(m, x, t, y)
        outs, pageable, graphs = res[name]
        print(f"{name}: pageable HtoD memcpys in one compiled forward {pageable}; unique graphs {graphs}; "
              f"repeat outputs identical {all(torch.equal(outs[0], o) for o in outs[1:])}")
    a7, a4 = res["C7"][0][-1], res["C4"][0][-1]
    print(f"C7 vs C4 output: identical {torch.equal(a7, a4)}, max abs diff {(a7 - a4).abs().max().item():.3e}, "
          f"max abs {a4.abs().max().item():.3e}")


if __name__ == "__main__":
    main()
