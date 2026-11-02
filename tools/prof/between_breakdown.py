"""CPU time inside one record_function window of a torch.profiler chrome trace, by Python frame.

python tools/prof/between_breakdown.py <rank0.json> <annotation name> [occurrence=1] [min_ms=0.5] [max_depth=6]
Lists the Python frames inside the window longer than min_ms, indented by nesting depth below the window.
"""
import json, sys

path, name = sys.argv[1], sys.argv[2]
occ = int(sys.argv[3]) if len(sys.argv) > 3 else 1
min_ms = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
max_depth = int(sys.argv[5]) if len(sys.argv) > 5 else 6
ev = json.load(open(path))["traceEvents"]
wins = sorted((e for e in ev if e.get("name") == name and "dur" in e and e.get("cat") in ("user_annotation", "cpu_op")),
              key=lambda e: e["ts"])
w = wins[occ]
lo, hi, tid = w["ts"], w["ts"] + w["dur"], w["tid"]
print(f"window {name} #{occ}: {w['dur']/1000:.1f} ms")
fr = sorted((e for e in ev if e.get("cat") in ("python_function", "cpu_op", "cuda_runtime") and e.get("tid") == tid and "dur" in e
             and e["ts"] >= lo and e["ts"] + e["dur"] <= hi), key=lambda e: (e["ts"], -e["dur"]))
stack = []
for e in fr:
    while stack and stack[-1] <= e["ts"]:
        stack.pop()
    depth = len(stack)
    stack.append(e["ts"] + e["dur"])
    if e["dur"] >= min_ms * 1000 and depth <= max_depth:
        print(f"{e['dur']/1000:8.2f} ms {'  ' * depth}{e['name'][:140]}")
