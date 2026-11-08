"""Paired halves (jobs/mfu/pairs.sh) compared only while both halves train in steady state: the summary's second halves
overlap the other half's compile and the time one half runs alone. usage: pair_window.py <pairs out dir>..."""
import glob, json, os, statistics as st, sys
for out in sys.argv[1:]:
    print("==", os.path.basename(out))
    for a_dir in sorted(glob.glob(os.path.join(out, "d*_A")), key=lambda p: int(os.path.basename(p).split("_")[0][1:])):
        tag = os.path.basename(a_dir)[:-2]
        ld = lambda h: json.load(open(os.path.join(out, f"{tag}_{h}", "bench", "steps_rank0.json")))["rows"]
        try:
            a, b = ld("A"), ld("B")
        except FileNotFoundError:
            continue
        ms = lambda x: x.get("step_ms", 1000 * (x["t_end"] - x["t_start"]))
        ss = lambda r: r[len(r) // 10:]           # steady state: after the first 10% of steps
        sa, sb = ss(a), ss(b)
        w0, w1 = max(sa[0]["t_start"], sb[0]["t_start"]), min(sa[-1]["t_end"], sb[-1]["t_end"])
        ina = [ms(x) for x in sa if w0 <= x["t_start"] <= w1]; inb = [ms(x) for x in sb if w0 <= x["t_start"] <= w1]
        h2 = lambda r: [ms(x) for x in r[len(r) // 2:]]
        if len(ina) < 50 or len(inb) < 50:
            print(f"{tag}: overlap {w1 - w0:.0f} s too short (A {len(ina)}, B {len(inb)} steps)"); continue
        print(f"{tag}: both in steady state for {w1 - w0:.0f} s (A {len(ina)} steps, B {len(inb)}): median A {st.median(ina):.1f} ms, "
              f"B {st.median(inb):.1f} ms ({st.median(inb) / st.median(ina) - 1:+.1%}); mean {st.mean(ina):.1f} vs {st.mean(inb):.1f} "
              f"({st.mean(inb) / st.mean(ina) - 1:+.1%}) | summary's second halves: median {st.median(h2(a)):.1f} vs {st.median(h2(b)):.1f} "
              f"({st.median(h2(b)) / st.median(h2(a)) - 1:+.1%})")
