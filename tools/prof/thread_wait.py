#!/usr/bin/env python3
"""CPU use and wait for a CPU of a training run's threads over some seconds, from /proc schedstat. Run on the job's node.
usage: thread_wait.py <rank to print threads of, or -1 for all ranks> <seconds>
Per rank: the four busiest threads with CPU %, wait %, involuntary context switches per second and CPUs allowed. Then the data
workers together: CPUs used and summed wait."""
import os
import subprocess
import sys
import time


def stat_of(path):
    """(on-CPU ns, wait ns) of a task directory."""
    with open(f"{path}/schedstat") as f:
        cpu, delay, _ = f.read().split()
    return int(cpu), int(delay)


def involuntary(path):
    with open(f"{path}/status") as f:
        for line in f:
            if line.startswith("nonvoluntary_ctxt_switches"):
                return int(line.split()[1])
    return 0


def local_rank(pid):
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            for item in f.read().split(b"\0"):
                if item.startswith(b"LOCAL_RANK="):
                    return int(item[11:])
    except (OSError, ValueError):
        pass
    return None


def parent(pid):
    with open(f"/proc/{pid}/stat") as f:
        return int(f.read().rsplit(")", 1)[1].split()[1])


def snapshot(pids):
    out = {}
    for pid in pids:
        try:
            for tid in os.listdir(f"/proc/{pid}/task"):
                path = f"/proc/{pid}/task/{tid}"
                try:
                    cpu, delay = stat_of(path)
                    with open(f"{path}/comm") as f:
                        comm = f.read().strip()
                    out[(pid, tid)] = (cpu, delay, involuntary(path), comm)
                except (OSError, ValueError):
                    pass
        except OSError:
            pass
    return out


def main():
    want, seconds = int(sys.argv[1]), float(sys.argv[2])
    found = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "main.py fit"], capture_output=True, text=True).stdout.split()
    pids = [int(p) for p in found]
    ranks = {p: local_rank(p) for p in pids}
    # a rank's main process has a parent outside its own rank; a data worker's parent is its rank's main process
    mains = {}
    for p in pids:
        try:
            pp = parent(p)
        except OSError:
            continue
        if ranks.get(pp, -2) != ranks[p]:
            mains[p] = ranks[p] if ranks[p] is not None else 0
    workers = [p for p in pids if p not in mains]
    t0 = time.time()
    a = snapshot(pids)
    time.sleep(seconds)
    b = snapshot(pids)
    dt = time.time() - t0
    print(f"{len(pids)} processes, {len(mains)} ranks, {dt:.1f} s, node {os.uname().nodename}")
    w_cpu = w_wait = 0
    for key, v in b.items():
        u = a.get(key)
        if u and key[0] in workers:
            w_cpu += v[0] - u[0]
            w_wait += v[1] - u[1]
    print(f"data workers: CPUs used {w_cpu / 1e9 / dt:.1f}, summed wait for a CPU {w_wait / 1e9 / dt:.1f} CPU-equivalents")
    for pid, rank in sorted(mains.items(), key=lambda x: x[1]):
        if want >= 0 and rank != want:
            continue
        rows = []
        for key, v in b.items():
            u = a.get(key)
            if u and key[0] == pid:
                try:
                    allowed = len(os.sched_getaffinity(int(key[1])))
                except OSError:
                    allowed = -1
                rows.append(((v[0] - u[0]) / 1e7 / dt, (v[1] - u[1]) / 1e7 / dt, (v[2] - u[2]) / dt, v[3], allowed))
        rows.sort(reverse=True)
        print(f"rank {rank} pid {pid}: " + "; ".join(f"{c} cpu {cp:.0f}% wait {w:.1f}% invol/s {nv:.0f} allowed {al}" for cp, w, nv, c, al in rows[:4]))


if __name__ == "__main__":
    main()
