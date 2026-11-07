"""Queue the ladder sweep: one resumable training job per size of docs/ladder_runs.json and one EMA FD eval per ms checkpoint.

python3 -I tools/submit_ladder.py <measured_ms_json> [--dry]   # measured_ms_json: {"d640": 57.1, ...} from the 8-GPU smoke
--time = 1.25 x (max_steps x step time + 15 min of start-ups and checkpoint saves), the rule for right-sized jobs.
"""
import json
import subprocess
import sys

WQ = "/scratch/gpfs/GROUP/USER/tools/wq/wq"
WT = "/scratch/gpfs/GROUP/USER/project/gpic-wt/ladder_run"
ms = json.loads(sys.argv[1])
dry = "--dry" in sys.argv
runs = json.load(open(f"{WT}/docs/ladder_runs.json"))["runs"]


def submit(*args):
    cmd = [WQ, "submit", "--project", "gpic-ladder", *args]
    print(" ".join(cmd), flush=True)
    return "w0" if dry else subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=WT).stdout.strip().split()[-1]


for r in runs:
    size, d, nb, nt, steps = r["size"], str(r["d"]), str(r["nb"]), str(r["nt"]), r["max_steps"]
    sec = int(1.25 * (steps * ms[size] / 1000 + 900))
    t = f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:00"
    w = submit("--partial", "--prio", "10", "--time", t, "-J", f"gpic-ladder-{size}", f"{WT}/jobs/train_ladder.sh", size, d, nb, nt, str(steps))
    ev = [submit("--after", f"ok:{w}", "--prio", "1", "-J", f"gpic-ladder-eval-{size}", f"{WT}/jobs/eval_ladder.sh", size, d, nb, nt, str(s))
          for s in (steps // 4, steps // 2, 3 * steps // 4, steps)]
    print(json.dumps({"size": size, "train": w, "time": t, "evals": ev}), flush=True)
