"""GPU health gate for benchmarks and timed training: every visible GPU runs a bf16 GEMM loop at the same time (as in
a DDP job, so a GPU that overheats under shared load shows it), then each GPU's throughput, clock and throttle reasons
are compared. Exit 1 if any GPU is below --min-frac of the median throughput or reports a thermal or hardware
slowdown; the job then exits 98 so wq requeues it (two nodes, node-k17g3 and node-k12g1, each had one GPU throttled
to ~60% clock on 2026-11-01, which slowed every DDP rank to its pace).

python tools/gpu_health.py [--seconds 8] [--min-frac 0.85] [--bad-nodes-file PATH]
Inside a Slurm job a failing node is also appended to tools/bad_nodes.txt (node, evidence, who, date; under a file
lock, once), so pilot_manager excludes it and no host there takes wq jobs (controller, 2026-11-01).
"""
import fcntl
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

BAD_REASONS = {0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}


def worker(seconds):
    import torch
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t, n, reasons = time.time(), 0, 0
    q = ["nvidia-smi", "-i", os.environ["HEALTH_GPU"], "--query-gpu=clocks.sm,temperature.gpu,clocks_event_reasons.active",
         "--format=csv,noheader,nounits"]
    samples = []
    while time.time() - t < seconds:
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()
        n += 20
        if time.time() - t > seconds / 2:   # sample under load, after the warm-up half
            out = subprocess.run(q, capture_output=True, text=True).stdout.strip().split(", ")
            samples.append((int(out[0]), int(out[1]), int(out[2], 16)))
            reasons |= int(out[2], 16)
    dt = time.time() - t
    print(json.dumps({"tflops": 2 * 8192 ** 3 * n / dt / 1e12, "clock": statistics.median(s[0] for s in samples),
                      "temp": max(s[1] for s in samples), "reasons": reasons}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--min-frac", type=float, default=0.85)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--bad-nodes-file", default="/scratch/gpfs/GROUP/USER/tools/bad_nodes.txt"
                    if os.environ.get("SLURM_JOB_ID") else "")
    a = ap.parse_args()
    if a.worker:
        return worker(a.seconds)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = vis.split(",") if vis else [str(i) for i in range(int(subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU ")))]
    procs = {}
    for g in gpus:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, HEALTH_GPU=g)
        procs[g] = subprocess.Popen([sys.executable, __file__, "--worker", "--seconds", str(a.seconds)], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    res = {}
    for g, p in procs.items():
        out, err = p.communicate()
        try:
            res[g] = json.loads(out.strip().splitlines()[-1])
        except Exception:
            res[g] = {"tflops": 0.0, "clock": 0, "temp": 0, "reasons": 0, "error": err[-300:]}
    med = statistics.median(r["tflops"] for r in res.values())
    bad = []
    for g, r in res.items():
        flags = [name for bit, name in BAD_REASONS.items() if r["reasons"] & bit]
        slow = r["tflops"] < a.min_frac * med
        print(f"GPU {g}: {r['tflops']:.0f} TFLOP/s, {r['clock']:.0f} MHz, {r['temp']} C, reasons 0x{r['reasons']:x} "
              f"{' '.join(flags)}{'  SLOW' if slow else ''}", flush=True)
        if slow or flags:
            bad.append(g)
    host = os.uname().nodename.split(".")[0]
    print(f"{host}: median {med:.0f} TFLOP/s; {'UNHEALTHY GPUs ' + ','.join(bad) if bad else 'healthy'}", flush=True)
    if bad and a.bad_nodes_file:
        ev = "; ".join(f"GPU {g} {res[g]['tflops']:.0f} TFLOP/s, {res[g]['clock']:.0f} MHz, {res[g]['temp']} C, "
                       f"reasons 0x{res[g]['reasons']:x}" for g in bad)
        line = (f"{host}   # {ev}; median {med:.0f} TFLOP/s under shared bf16 GEMM load (gpu_health.py in job "
                f"{os.environ.get('FETCHED_JOB_ID') or os.environ.get('SLURM_JOB_ID')}, genmodel, "
                f"{time.strftime('%Y-%m-%d %H:%M')})\n")
        with open(a.bad_nodes_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            listed = {ln.split()[0] for ln in f if ln.strip() and not ln.startswith("#")}
            if host not in listed:
                f.write(line)
                print(f"appended {host} to {a.bad_nodes_file}", flush=True)
            fcntl.flock(f, fcntl.LOCK_UN)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
