"""Text encoder outputs under different attention implementations, as the conditioner calls it (300-token max_length
padding, attention mask, last hidden state, bf16). Reference: SDPA (the official path). Every other variant is compared
on valid and on padded positions (the denoiser reads all 300)."""
import json, torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import Qwen3Model, Qwen2Tokenizer
caps = [json.loads(l)["caption"] for l, _ in zip(open("/scratch/gpfs/GROUP/USER/data/gpic/gpic_test_50k.jsonl"), range(16))]
tok = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-1.7B", max_length=300, padding_side="right")
t = tok(caps, truncation=True, max_length=300, padding="max_length", return_tensors="pt")
ids, am = t.input_ids.cuda(), t.attention_mask.cuda()
out = {}
def run(impl, backend=None, dtype=torch.bfloat16):
    m = Qwen3Model.from_pretrained("Qwen/Qwen3-1.7B", attn_implementation=impl).to(dtype).cuda().eval()
    with torch.no_grad():
        if backend is None:
            r = m(input_ids=ids, attention_mask=am)[0].float()
        else:
            with sdpa_kernel(backend):
                r = m(input_ids=ids, attention_mask=am)[0].float()
    del m; torch.cuda.empty_cache(); return r
out["sdpa (official, bf16)"] = run("sdpa")
out["sdpa math backend, bf16"] = run("sdpa", SDPBackend.MATH)
out["eager, bf16"] = run("eager")
out["flex, bf16"] = run("flex_attention")
out["sdpa, fp32"] = run("sdpa", None, torch.float32)
out["eager, fp32"] = run("eager", None, torch.float32)
ref = out["sdpa (official, bf16)"]; valid = am.bool()
for k, v in out.items():
    d = (v - ref).abs()
    print(f"{k:28s} vs official: valid max {d[valid].max():7.3f} mean {d[valid].mean():.4f} | pad max {d[~valid].max():7.3f} mean {d[~valid].mean():.4f}")
d = (out["eager, fp32"] - out["sdpa, fp32"]).abs()
print(f"fp32 eager vs fp32 sdpa: valid max {d[valid].max():.4f} | pad max {d[~valid].max():.4f}")
