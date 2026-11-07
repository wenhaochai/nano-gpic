"""The C++ kernel launcher computes the same math as the Python launcher, up to bf16 rounding: one bf16 training step of a small
JiT_T2I, compiled as in training, under TORCHINDUCTOR_CPP_WRAPPER=0 and 1. The results are not bit-identical (2026-11-03, A40:
0.25% of the output entries differ, by one bf16 ulp; gradient relative L2 median 3.3e-4, max 2.1e-3, against 1e-6 and 3.6e-4 for
one launcher twice). Pass: output within two ulps at its scale, gradient relative L2 at most 5e-3. On a vis node with a GPU:
  PYTHONPATH=baselines/PixelGen python tools/test_cpp_launcher.py <dir>
Runs itself once per setting, twice each, in separate processes.
"""
import os
import subprocess
import sys


def one_run(path):
    import torch
    from src.models.transformer.JiT_T2I import JiT_T2I
    torch.manual_seed(0)
    net = JiT_T2I(input_size=256, in_channels=3, num_groups=4, hidden_size=256, num_blocks=2, num_text_blocks=1, patch_size=16,
                  txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=32).cuda()
    for p in net.parameters():
        p.data.add_(torch.randn_like(p) * 0.02)
    net.compile()
    B, T = 32, 64
    x = torch.randn(B, 3, 256, 256, device="cuda")
    t = torch.rand(B, device="cuda")
    y = torch.randn(B, T, 2048, device="cuda")
    mask = torch.zeros(B, T, dtype=torch.bool, device="cuda")
    mask[:, :T // 2] = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = net(x, t, y, y_mask=mask)
        loss = out.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    torch.save({"out": out.float().cpu(), "loss": loss.item(),
                "grads": {k: p.grad.float().cpu() for k, p in net.named_parameters() if p.grad is not None}}, path)


def main():
    if len(sys.argv) > 2:
        one_run(sys.argv[2])
        return
    import torch
    d = sys.argv[1]
    os.makedirs(d, exist_ok=True)
    runs = {}
    for name, value in (("py0", "0"), ("py1", "0"), ("cpp0", "1"), ("cpp1", "1")):
        path = os.path.join(d, name + ".pt")
        env = {**os.environ, "TORCHINDUCTOR_CPP_WRAPPER": value}
        r = subprocess.run([sys.executable, __file__, "run", path], env=env, capture_output=True, text=True)
        if r.returncode:
            print(name, "failed:", r.stderr[-1500:])
            sys.exit(1)
        runs[name] = torch.load(path, weights_only=False)

    def diff(a, b):
        o = (a["out"] - b["out"]).abs().max().item()
        rel = [((a["grads"][k] - b["grads"][k]).norm() / a["grads"][k].norm().clamp_min(1e-30)).item() for k in a["grads"]]
        return o, max(rel), abs(a["loss"] - b["loss"])

    print("params with grad", len(runs["py0"]["grads"]), "loss", runs["py0"]["loss"])
    print("same launcher twice: out max diff, grad relative L2 max, loss diff", diff(runs["py0"], runs["py1"]), diff(runs["cpp0"], runs["cpp1"]))
    across = [diff(runs["py0"], runs["cpp0"]), diff(runs["py1"], runs["cpp1"])]
    print("python vs C++:       out max diff, grad relative L2 max, loss diff", across)
    scale = runs["py0"]["out"].abs().max().item()
    ok = all(a[0] <= 2 * 2 ** (torch.tensor(scale).log2().floor().item() - 7) and a[1] <= 5e-3 for a in across)
    print("ALL OK" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
