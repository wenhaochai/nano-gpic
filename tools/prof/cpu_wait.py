#!/usr/bin/env python3
"""Are threads waiting for a CPU while other CPUs are idle? Per CPU, from /proc/schedstat (rq_cpu_time, rq_run_delay; no privilege), and per rank, the data workers' CPU use
and wait from /proc/<pid>/task/*/schedstat. Run on the job's node during a training run.
usage: cpu_wait.py <seconds> [interval s]
Prints: per interval the number of idle CPUs (under 50% busy) and the summed wait on each socket, as the means over the window and the share of intervals with both at least 4 idle
CPUs and at least 2 CPU-equivalents of wait on one socket; per CPU the busy share and wait; per rank the workers' CPU use and wait."""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thread_wait import local_rank, parent, snapshot  # noqa: E402


def read_cpus():
    out = {}
    with open("/proc/schedstat") as f:
        for line in f:
            if line.startswith("cpu"):
                p = line.split()
                out[int(p[0][3:])] = (int(p[7]), int(p[8]))
    return out


def node_cpus():
    nodes = {}
    for n in sorted(os.listdir("/sys/devices/system/node")):
        if n.startswith("node"):
            with open(f"/sys/devices/system/node/{n}/cpulist") as f:
                cpus = []
                for part in f.read().strip().split(","):
                    a, _, b = part.partition("-")
                    cpus.extend(range(int(a), int(b or a) + 1))
            nodes[int(n[4:])] = cpus
    return nodes


def main():
    seconds = float(sys.argv[1])
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    pids = [int(p) for p in subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "main.py fit"], capture_output=True, text=True).stdout.split()]
    ranks = {p: local_rank(p) for p in pids}
    mains = {}
    for p in pids:
        try:
            if ranks.get(parent(p), -2) != ranks[p]:
                mains[p] = ranks[p] if ranks[p] is not None else 0
        except OSError:
            pass
    owner = {}
    for p in pids:
        if p not in mains:
            try:
                owner[p] = mains.get(parent(p))
            except OSError:
                pass
    nodes = node_cpus()
    a0 = snapshot(pids)
    t_start = time.time()
    first = prev = read_cpus()
    t_prev = t_start
    rows = []
    while time.time() - t_start < seconds:
        time.sleep(dt)
        now, t_now = read_cpus(), time.time()
        span = (t_now - t_prev) * 1e9
        rows.append({c: ((now[c][0] - prev[c][0]) / span, (now[c][1] - prev[c][1]) / span) for c in now})
        prev, t_prev = now, t_now
    total = (t_prev - t_start) * 1e9
    b0 = snapshot(pids)
    print(f"{len(pids)} processes, {len(mains)} ranks, {t_prev - t_start:.1f} s, {len(rows)} intervals, node {os.uname().nodename}")
    for n, cpus in nodes.items():
        idle = [sum(r[c][0] < 0.5 for c in cpus) for r in rows]
        wait = [sum(r[c][1] for c in cpus) for r in rows]
        both = sum(i >= 4 and w >= 2 for i, w in zip(idle, wait)) / len(rows)
        print(f"socket {n}: {len(cpus)} CPUs, idle CPUs mean {sum(idle) / len(rows):.1f}, summed wait mean {sum(wait) / len(rows):.1f} CPU-equivalents, share of intervals with >=4 idle CPUs and >=2 of wait {both:.3f}")
    print("cpu busy% wait%: " + " ".join(f"{c}:{100 * (prev[c][0] - first[c][0]) / total:.0f}/{100 * (prev[c][1] - first[c][1]) / total:.0f}" for c in sorted(prev)))
    per = {}
    for key, v in b0.items():
        u = a0.get(key)
        if u and key[0] in owner:
            r = per.setdefault(owner[key[0]], [0, 0])
            r[0] += v[0] - u[0]
            r[1] += v[1] - u[1]
    for r in sorted(per, key=lambda x: (x is None, x)):
        print(f"rank {r} workers: CPUs used {per[r][0] / total:.1f}, summed wait {per[r][1] / total:.1f} CPU-equivalents")


if __name__ == "__main__":
    main()
