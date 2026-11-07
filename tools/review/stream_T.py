"""Mean padded T under the real stream order (one worker reads a shard front to back, wds.shuffle(100), windows of
1024, sorted, chunks of 32, pad to 64-multiple <= 320) vs the global shuffle that tools/ladder_flops.py used.
Also: caption_type runs in tar order."""
import json, random, sys, tarfile
from collections import Counter
from transformers import Qwen2Tokenizer
tok = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-1.7B", padding_side="right")
def padT(lens):
    Ts = []
    for w in range(0, len(lens) - 1023, 1024):
        win = sorted(lens[w:w + 1024])
        Ts += [min(-(-max(win[c:c + 32]) // 64) * 64, 320) for c in range(0, 1024, 32)]
    return Ts
def shuf100(xs, rng):
    buf, out = [], []
    for x in xs:
        buf.append(x)
        if len(buf) >= 100:
            k = rng.randrange(len(buf)); buf[k], buf[-1] = buf[-1], buf[k]; out.append(buf.pop())
    rng.shuffle(buf); return out + buf
rng = random.Random(0); allT_stream, all_lens = [], []
for i in [int(s) for s in sys.argv[1:]]:
    with tarfile.open(f"/scratch/gpfs/GROUP/USER/data/gpic/train/gpic_train_{i:05d}.tar") as t:
        metas = [json.loads(t.extractfile(m).read()) for m in t if m.name.endswith(".json")]
    caps = [m["caption"] for m in metas]; types = [m["caption_type"] for m in metas]
    lens = [min(len(x), 320) for x in tok(caps, truncation=True, max_length=320)["input_ids"]]
    runs = sum(1 for a, b in zip(types, types[1:]) if a != b) + 1
    Ts = padT(shuf100(lens, rng)); allT_stream += Ts; all_lens += lens
    print(f"shard {i}: n {len(lens)} types {dict(Counter(types))} type-runs {runs} stream mean T {sum(Ts)/len(Ts):.1f}")
g = list(all_lens); random.Random(0).shuffle(g); Tg = padT(g)
print(f"stream-order mean T {sum(allT_stream)/len(allT_stream):.1f} hist {dict(sorted(Counter(allT_stream).items()))}")
print(f"global-shuffle mean T {sum(Tg)/len(Tg):.1f} hist {dict(sorted(Counter(Tg).items()))}")
