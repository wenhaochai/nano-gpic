#!/usr/bin/env python3
"""Held-out loss of the released GPIC model (stanford-vision-lab/gpic-baselines, step 390600; owner popup 2026-11-08 23:5x, REL-HO),
on the ladder's held-out metric: the same 3,072 validation images and 3,072 images of train shard 00787, the same 16 levels and the same
fixed noise as tools/heldout_loss.py, its EMA weights (ema_denoiser.*) and its raw weights (denoiser.*).

The one difference is the text path. The ladder trains on the masked path (real-token Qwen3-1.7B outputs, a padding mask, at most 320
tokens), and tools/heldout_loss.py scores it on that path. The released model was trained on the official path: captions tokenized,
cut or padded to 300 tokens, Qwen3-1.7B run with the attention mask, every one of the 300 output positions handed to the denoiser with
no mask (Qwen3TextEncoder text_mode "official", JiT_T2I y_mask None, txt_max_length 300). Scoring it on the masked path would measure it
on inputs it never saw, so this script encodes the captions the official way and scores with y_mask None.

  python tools/heldout_release.py work --weights ema --split val --gpu 0 --out part.json   (one process per weights and split)
  python tools/heldout_release.py merge part.json ...   (one line per weights appended to tmp/heldout_loss/heldout_loss.jsonl)
--limit and --levels shrink a run for a CPU smoke test (--gpu -1)."""
import argparse, json, math, os, socket, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heldout_loss as hl  # noqa: E402

CKPT = os.path.join(hl.WC, "models", "gpic-baselines", "baseline-jit-t2i-gpic-full-step=390600.ckpt")
STEP = 390600
KW = dict(input_size=hl.IMAGE, in_channels=3, num_groups=24, hidden_size=1536, num_blocks=16, num_text_blocks=4, patch_size=16,
          txt_embed_dim=hl.EMB_DIM, txt_max_length=300, bottleneck_dim=256)   # the release evaluation's model (release-eval 767e9b6)
LOSS = dict(timeshift=1.0, P_mean=-0.8, P_std=0.8, t_eps=0.05)   # the ladder's loss settings: the same 16 levels as every record
TXT = 300
# the page's FLOPs for the released model: 390,600 steps at batch 256, 9.77 times the 1.4258e19-FLOP budget (figures.js)
TRAIN_FLOPS, FLOPS_SOURCE = 9.77 * 1.4258e19, "the page's FLOPs: 390,600 steps x batch 256 = 9.77 x 1.4258e19 (figures.js)"
OUT = os.path.join(hl.OUT_DIR, "heldout_loss.jsonl")


def encode_official(captions, device, batch=32):
    """Qwen3TextEncoder's official branch (_impl_condition, text_mode "official"), without its CUDA-only pinned copies: the captions
    cut or padded to 300 tokens on the right, Qwen3-1.7B in bf16 with the attention mask, all 300 outputs kept."""
    from transformers import Qwen2Tokenizer, Qwen3Model
    tok = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-1.7B", max_length=TXT, padding_side="right")
    model = Qwen3Model.from_pretrained("Qwen/Qwen3-1.7B", attn_implementation="sdpa").to(torch.bfloat16).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(captions), batch):
            t = tok(captions[i:i + batch], truncation=True, max_length=TXT, padding="max_length", return_tensors="pt")
            y = model(input_ids=t.input_ids.to(device), attention_mask=t.attention_mask.to(device))[0]
            assert y.shape[1:] == (TXT, hl.EMB_DIM), y.shape   # Qwen3-1.7B's width is the denoiser's txt_embed_dim
            out.append(y)
    del model
    return torch.cat(out)


def official_text_batch(split, idx, device):
    y = split["y_official"]
    y = y[torch.tensor(idx, device=y.device)].to(device)
    if torch.device(device).type == "cpu":   # the CPU smoke runs without autocast; on a GPU bf16 goes in under autocast, as in the ladder's records
        y = y.float()
    return y, None   # y_mask None: the official unmasked path


def cmd_work(a):
    device = "cpu" if a.gpu < 0 else f"cuda:{a.gpu}"
    if a.gpu >= 0:
        torch.cuda.set_device(a.gpu)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    data = hl.load_data(hl.data_path(a.n_val, a.n_train), device)
    s = data[a.split]
    n = len(s["lens"]) if a.limit is None else a.limit
    split = dict(img=s["img"][:n], lens=s["lens"][:n], order=[i for i in s["order"] if i < n])
    split["y_official"] = encode_official(s["captions"][:n], device)
    enc_s = time.time() - t0
    ck = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    assert ck["global_step"] == STEP, ck["global_step"]
    net = hl.build_denoiser(KW, ck["state_dict"], dict(hl.WEIGHTS)[a.weights], device)   # strict: a wrong shape fails here
    del ck
    tr = hl.make_trainer(LOSS)
    ts = hl.levels(LOSS["P_mean"], LOSS["P_std"], LOSS["timeshift"])
    if a.levels:
        ts = ts[:a.levels]
    load_s = time.time() - t0
    hl.text_batch = official_text_batch   # evaluate() looks the function up at call time
    res = hl.evaluate({a.weights: net}, tr, split, ts, hl.SEED, a.batch, device, device != "cpu", 4,
                      log=lambda m: print(f"[release] {a.weights} {a.split}: {m}", flush=True))[a.weights]
    lm, im = res["level_mean"], res["image_mean"]
    part = dict(weights=a.weights, split=a.split, n=n, t=ts.tolist(), loss=float(lm.mean()), loss_t=lm.tolist(),
                se=float(im.std() / math.sqrt(len(im))) if n > 1 else None, data=data["tag"],
                train_shard=os.path.basename(data["train_shard"]), elapsed_s=round(time.time() - t0, 1),
                encode_s=round(enc_s, 1), load_s=round(load_s, 1), gpu=torch.cuda.get_device_name(device) if a.gpu >= 0 else "cpu",
                host=socket.gethostname(), job=os.environ.get("FETCHED_JOB_ID") or os.environ.get("SLURM_JOB_ID", ""))
    with open(a.out, "w") as f:
        json.dump(part, f)
    print(f"[release] {a.weights} {a.split}: loss {part['loss']:.5f} over {n} images and {len(ts)} levels in {part['elapsed_s']:.0f} s",
          flush=True)


def cmd_merge(a):
    parts = [json.load(open(p)) for p in a.parts]
    by = {(p["weights"], p["split"]): p for p in parts}
    ref = None
    with open(OUT) as f:   # the levels must equal the ladder records' levels
        for line in f:
            ref = json.loads(line)["t"]
            break
    recs = []
    for w in ("ema", "raw"):
        v, tr = by[(w, "val")], by[(w, "train")]
        assert v["n"] == a.n_val and tr["n"] == a.n_train, (v["n"], tr["n"])
        assert max(abs(x - y) for x, y in zip(v["t"], ref)) < 1e-7 and len(v["t"]) == len(ref) == hl.NUM_T, "levels differ"
        recs.append(dict(ckpt=CKPT, run_root=os.path.dirname(CKPT), ladder="release", size="d1536", step=STEP, d=1536, nb=16, nt=4,
                         train_flops=TRAIN_FLOPS, flops_source=FLOPS_SOURCE, weights=w, n_val=a.n_val, n_train=a.n_train,
                         num_t=hl.NUM_T, seed=hl.SEED, data=v["data"], train_shard=v["train_shard"], t=v["t"],
                         loss_val=v["loss"], loss_val_t=v["loss_t"], loss_val_se=v["se"],
                         loss_train=tr["loss"], loss_train_t=tr["loss_t"], loss_train_se=tr["se"],
                         text="official: captions cut or padded to 300 tokens, every output position, no mask (the released model's "
                              "training path); the ladder's records use the masked path",
                         elapsed_s=max(v["elapsed_s"], tr["elapsed_s"]), gpu=v["gpu"], host=v["host"], job=v["job"],
                         code=hl._git_head(), tool_sha1=hl._sha1(os.path.abspath(__file__)), time=time.strftime("%Y-%m-%dT%H:%M:%S")))
    if a.dry:
        print(json.dumps(recs, indent=1)[:3000])
        return
    hl.append_lines(OUT, recs)
    print(" ".join(f"{r['weights']}: val {r['loss_val']:.5f} train {r['loss_train']:.5f}" for r in recs), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("work")
    w.add_argument("--weights", choices=("ema", "raw"), required=True)
    w.add_argument("--split", choices=("val", "train"), required=True)
    w.add_argument("--gpu", type=int, default=0)
    w.add_argument("--out", required=True)
    w.add_argument("--batch", type=int, default=hl.BATCH)
    w.add_argument("--threads", type=int, default=4)
    w.add_argument("--limit", type=int, default=None)
    w.add_argument("--levels", type=int, default=None)
    m = sub.add_parser("merge")
    m.add_argument("parts", nargs="+")
    m.add_argument("--dry", action="store_true")
    for p in (w, m):
        p.add_argument("--n-val", type=int, default=3072)
        p.add_argument("--n-train", type=int, default=3072)
    a = ap.parse_args()
    {"work": cmd_work, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
