#!/usr/bin/env python3
"""Are data workers waiting for a CPU while CPUs they may use are idle? Run on the job's node during a training run.
usage: cpu_wait.py <seconds> [interval s]
Per interval, per socket: idle CPUs among those the workers may use, from the idle and iowait ticks of /proc/stat (rq_cpu_time would count a long run only when it ends, docs/review_pin2_2026-11-04.md F1),
and the workers in state R by last CPU, from /proc/<pid>/stat. Printed: the means, and the share of intervals with at least 4 such idle CPUs and more workers in state R than busy CPUs.
Over the whole window: per CPU busy and wait from /proc/schedstat; per rank the workers' CPU use and wait; the workers with the largest wait, summed over their threads."""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thread_wait import local_rank, parent, snapshot  # noqa: E402


def read_schedstat():
    out = {}
    with open("/proc/schedstat") as f:
        for line in f:
            if line.startswith("cpu"):
                p = line.split()
                out[int(p[0][3:])] = (int(p[7]), int(p[8]))
    return out


def read_stat():
    """{cpu: (idle + iowait ticks, all ticks)}, procs_running."""
    cpus, running = {}, 0
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                v = [int(x) for x in p[1:9]]
                cpus[int(p[0][3:])] = (v[3] + v[4], sum(v))
            elif line.startswith("procs_running"):
                running = int(line.split()[1])
    return cpus, running


def worker_states(pids):
    """{pid: (state, last CPU)} of each worker's main thread."""
    out = {}
    for p in pids:
        try:
            with open(f"/proc/{p}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            out[p] = (fields[0], int(fields[36]))
        except (OSError, ValueError, IndexError):
            pass
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
    workers = list(owner)
    allowed = {}
    for p in workers:
        try:
            allowed[p] = set(os.sched_getaffinity(p))
        except OSError:
            allowed[p] = set()
    nodes = node_cpus()
    socket_of = {c: n for n, cpus in nodes.items() for c in cpus}
    usable = {n: set().union(*[allowed[p] for p in workers if allowed[p] and allowed[p] <= set(cpus)] or [set()]) for n, cpus in nodes.items()}
    a0 = snapshot(pids)
    first = read_schedstat()
    t_start = time.time()
    prev, _ = read_stat()
    rows, running = [], []
    while time.time() - t_start < seconds:
        time.sleep(dt)
        now, run = read_stat()
        states = worker_states(workers)
        row = {}
        for n, cpus in nodes.items():
            idle = busy = 0
            for c in cpus:
                ticks = now[c][1] - prev[c][1]
                share = (now[c][0] - prev[c][0]) / ticks if ticks else 1.0
                if share > 0.5:
                    idle += c in usable[n]
                else:
                    busy += 1
            row[n] = (idle, busy, sum(1 for s, c in states.values() if s == "R" and socket_of.get(c) == n))
        rows.append(row)
        running.append(run)
        prev = now
    t_end = time.time()
    total = (t_end - t_start) * 1e9
    last = read_schedstat()
    b0 = snapshot(pids)
    print(f"{len(pids)} processes, {len(mains)} ranks, {t_end - t_start:.1f} s, {len(rows)} intervals, node {os.uname().nodename}, procs_running mean {sum(running) / len(running):.1f}")
    for n in nodes:
        idle = [r[n][0] for r in rows]
        busy = [r[n][1] for r in rows]
        rw = [r[n][2] for r in rows]
        both = sum(i >= 4 and w > b for i, b, w in zip(idle, busy, rw)) / len(rows)
        print(f"socket {n}: {len(usable[n])} CPUs the workers may use, idle among them mean {sum(idle) / len(rows):.1f}, busy CPUs mean {sum(busy) / len(rows):.1f}, workers in state R mean {sum(rw) / len(rows):.1f}, "
              f"share of intervals with >=4 idle and more workers in R than busy CPUs {both:.3f}")
    print("cpu busy% wait%: " + " ".join(f"{c}:{100 * (last[c][0] - first[c][0]) / total:.0f}/{100 * (last[c][1] - first[c][1]) / total:.0f}" for c in sorted(last)))
    per, each = {}, {}
    for key, v in b0.items():
        u = a0.get(key)
        if u and key[0] in owner:
            r = per.setdefault(owner[key[0]], [0, 0])
            r[0] += v[0] - u[0]
            r[1] += v[1] - u[1]
            e = each.setdefault(key[0], [0, 0])
            e[0] += v[0] - u[0]
            e[1] += v[1] - u[1]
    for r in sorted(per, key=lambda x: (x is None, x)):
        print(f"rank {r} workers: CPUs used {per[r][0] / total:.1f}, summed wait {per[r][1] / total:.1f} CPU-equivalents")
    for p, e in sorted(each.items(), key=lambda x: -x[1][1])[:8]:
        print(f"worker pid {p} rank {owner[p]}: cpu {100 * e[0] / total:.0f}%, wait {100 * e[1] / total:.0f}%, allowed {len(allowed.get(p, ()))}")
    vals = sorted(100 * e[1] / total for e in each.values())
    if vals:
        print(f"workers' wait %: median {vals[len(vals) // 2]:.1f}, p90 {vals[int(len(vals) * 0.9)]:.1f}, max {vals[-1]:.1f}")


if __name__ == "__main__":
    main()
