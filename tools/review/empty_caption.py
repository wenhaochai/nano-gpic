"""(1) Count captions with 0 tokens / > 320 tokens in a few training shards (json members only).
(2) Denoiser behaviour when one sample's y_mask row is all False (a 0-token caption that CFG did not drop)."""
import json, sys, tarfile, torch
from transformers import Qwen2Tokenizer
tok = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-1.7B", padding_side="right")
tot = zero = long_ = 0; examples = []
for i in [int(s) for s in sys.argv[1:]]:
    with tarfile.open(f"/scratch/gpfs/GROUP/USER/data/gpic/train/gpic_train_{i:05d}.tar") as t:
        for m in t:
            if not m.name.endswith(".json"): continue
            meta = json.loads(t.extractfile(m).read())
            if "caption" not in meta or "caption_type" not in meta: continue
            c = meta["caption"]
            if not isinstance(c, str): examples.append(("nonstr", m.name, repr(c)[:40])); continue
            n = len(tok(c)["input_ids"]); tot += 1
            if n == 0: zero += 1; examples.append(("zero", m.name, repr(c)))
            if n > 320: long_ += 1
print(f"captions {tot}  zero-token {zero}  >320 tokens {long_}  examples {examples[:5]}")

from src.models.transformer.JiT_T2I import JiT_T2I
torch.manual_seed(0)
m = JiT_T2I(input_size=64, in_channels=3, num_groups=4, hidden_size=256, num_blocks=2, num_text_blocks=1, patch_size=16,
            txt_embed_dim=64, txt_max_length=320, bottleneck_dim=32)
for p in m.parameters():   # non-zero adaLN / final layer so the output depends on every block
    p.data.add_(torch.randn_like(p) * 0.02)
x = torch.randn(2, 3, 64, 64); y = torch.randn(2, 64, 64); t = torch.rand(2)
mask = torch.zeros(2, 64, dtype=torch.bool); mask[0, :10] = True   # sample 1: zero valid tokens
out = m(x, t, y, y_mask=mask)
print("sample0 finite:", torch.isfinite(out[0]).all().item(), " sample1 (all-False mask) finite:", torch.isfinite(out[1]).all().item())
loss = out.pow(2).mean(); loss.backward()
print("loss finite:", torch.isfinite(loss).item(), " any NaN grad:", any((~torch.isfinite(p.grad)).any().item() for p in m.parameters() if p.grad is not None))
