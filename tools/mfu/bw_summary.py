#!/usr/bin/env python3
"""Summary of Inductor's per-kernel bandwidth profile (TORCHINDUCTOR_PROFILE=1, TORCHINDUCTOR_PROFILE_OUTPUT=<file>).

The file holds one table per call of each compiled graph (wrapper file): every Triton kernel's time (benchmarked
alone), bytes moved (Inductor's estimate from the kernel's inputs and outputs) and GB/s. Takes the last table of each
graph (steady state), then lists the kernels by excess time = time - bytes / PEAK_BW (the time above what the bytes need at
the reference bandwidth), with the graph each kernel is in.
usage: python bw_summary.py <file> [--bw 3000] [--top 40]
"""
import argparse, collections, re

ROW = re.compile(r"^\s*([\d.]+)ms\s+([\d.]+)\s*GB\s+([\d.]+)GB/s\s+([\d.]+)%\s+(\S+)")
HEAD = re.compile(r"TRITON KERNELS BANDWIDTH INFO \((.*)\)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--bw", type=float, default=3000.0, help="reference bandwidth, GB/s")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    tables, cur = collections.OrderedDict(), None
    for line in open(a.file):
        m = HEAD.search(line)
        if m:
            cur = m.group(1)
            tables[cur] = []   # a later call of the same graph replaces the earlier table
            continue
        m = ROW.match(line)
        if m and cur is not None:
            ms, gb, gbs, pct, name = m.groups()
            tables[cur].append((float(ms), float(gb), float(gbs), name))
    allk = []
    print(f"{len(tables)} graphs")
    for g, rows in tables.items():
        t, b = sum(r[0] for r in rows), sum(r[1] for r in rows)
        print(f"  {t:8.3f} ms {b:8.3f} GB {b / (t / 1e3) if t else 0:8.1f} GB/s  {len(rows):4d} kernels  {g.split('/')[-1]}")
        allk += [(ms, gb, gbs, name, g.split("/")[-1][:12]) for ms, gb, gbs, name in rows]
    T, B = sum(k[0] for k in allk), sum(k[1] for k in allk)
    ideal = B / a.bw * 1e3
    print(f"all Triton kernels: {T:.3f} ms, {B:.3f} GB, {B / (T / 1e3):.1f} GB/s; at {a.bw:.0f} GB/s the bytes need {ideal:.3f} ms "
          f"(excess {T - ideal:.3f} ms)")
    print(f"\ntop {a.top} by excess time (ms, GB, GB/s, excess ms, kernel, graph):")
    for ms, gb, gbs, name, g in sorted(allk, key=lambda k: -(k[0] - k[1] / a.bw * 1e3))[:a.top]:
        print(f"  {ms:7.3f} {gb:7.3f} {gbs:8.1f} {ms - gb / a.bw * 1e3:7.3f}  {name:60s} {g}")


if __name__ == "__main__":
    main()
