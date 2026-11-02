"""Python stacks behind long cudaStreamSynchronize calls in a torch.profiler chrome trace (with_stack=True).

python tools/prof/sync_stacks.py <rank0.json> [min_ms=5] [frames=14]
For each synchronize longer than min_ms: its duration, the memcpy launched just before it on the same thread, and the
innermost Python frames that enclose it.
"""
import json, sys

path = sys.argv[1]; min_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 5; nfr = int(sys.argv[3]) if len(sys.argv) > 3 else 14
ev = json.load(open(path))["traceEvents"]
py = [e for e in ev if e.get("cat") == "python_function" and "dur" in e]
rt = [e for e in ev if e.get("cat") == "cuda_runtime" and "dur" in e]
syncs = [e for e in rt if "Synchronize" in e["name"] and e["dur"] >= min_ms * 1000]
memcpys = sorted((e for e in rt if e["name"].startswith("cudaMemcpy")), key=lambda e: e["ts"])
for s in sorted(syncs, key=lambda e: e["ts"]):
    prev = [m for m in memcpys if m["tid"] == s["tid"] and m["ts"] <= s["ts"]]
    enc = sorted((p for p in py if p["tid"] == s["tid"] and p["ts"] <= s["ts"] and p["ts"] + p["dur"] >= s["ts"] + s["dur"]),
                 key=lambda p: p["ts"])
    print(f"== {s['name']} {s['dur']/1000:.1f} ms; last memcpy args {prev[-1].get('args', {}) if prev else None}")
    for p in enc[-nfr:]:
        print("   ", p["name"][:160])
