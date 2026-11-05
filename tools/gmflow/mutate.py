#!/usr/bin/env python3
"""Mutation tests of tools/gmflow/test_gmflow.py: each planted bug goes into a temporary copy of baselines/PixelGen/src, the tests run on the copy, and at least one
must fail. The worktree is never edited. Run on a vis node: tools/gmflow/run_vis.sh tools/gmflow/mutate.py [name ...]"""
import os
import shutil
import subprocess
import sys
import tempfile

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(WT, "baselines/PixelGen/src")
F = "diffusion/flow_matching/"
MUTANTS = {
    "softmax_over_channels": ("models/transformer/JiT_T2I.py", "logweights = logits.float().log_softmax(dim=1)", "logweights = logits.float().log_softmax(dim=2)"),
    "c1_c2_swapped": (F + "gmflow.py", "    return c1, c2, c3\n", "    return c2, c1, c3\n"),
    "c3_dropped": (F + "gmflow.py", "    c3 = beta_over_sigma_sq * sigma_to ** 2\n", "    c3 = 0 * beta_over_sigma_sq * sigma_to ** 2\n"),
    "s_x_equals_s_u": (F + "gmflow.py", '    return gm["logstd"] + torch.log(sigma.clamp(min=EPS))\n', '    return gm["logstd"] + 0 * sigma\n'),
    "no_bridge_noise": (F + "sampling_gmflow.py", "c3.clamp(min=0).sqrt() * torch.randn_like(x)", "0 * torch.randn_like(x)"),
}


def main():
    names = sys.argv[1:] or list(MUTANTS)
    survived = []
    for name in names:
        path, old, new = MUTANTS[name]
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            shutil.copytree(SRC, os.path.join(tmp, "src"))
            f = os.path.join(tmp, "src", path)
            text = open(f).read()
            assert text.count(old) == 1, (name, "pattern not found once")
            open(f, "w").write(text.replace(old, new))
            env = dict(os.environ, PYTHONPATH=tmp)
            r = subprocess.run([sys.executable, os.path.join(WT, "tools/gmflow/test_gmflow.py")], env=env, cwd=tmp, capture_output=True, text=True, timeout=1500)
            failed = [l.split()[0] for l in r.stdout.splitlines() if l.endswith(" FAIL")]
            print(f"{name}: {path}: '{old.strip()}' -> '{new.strip()}': failing tests {failed or 'NONE'}", flush=True)
            if not failed:
                survived.append(name)
    print("SURVIVED: " + " ".join(survived) if survived else "every mutant killed")
    sys.exit(len(survived))


if __name__ == "__main__":
    main()
