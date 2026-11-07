"""CPU cost of every stage of the training data path, per sample and per batch (one process, no GPU).

Stages, as the job runs them (src/data/dataset/gpic_dataset.py, augmentation.py, qwen3_text_encoder.py):
  tar member read -> json.loads -> PIL decode (open + convert RGB) -> random_crop_arr(256) -> ToTensor -> Normalize
  -> collate (stack 32) -> pin_memory ; tokenizer on 32 captions padded to 300 tokens (main process, every step).
Also: the end-to-end webdataset pipeline (GPICWebDataset.create_dataset) on one shard, one worker, to see the
framework overhead above the sum of stages; and the source image size distribution (decode cost scales with it).
usage: python datapath_prof.py <tar> <n> <out.json>
"""
import io, json, statistics, sys, tarfile, time

import numpy as np
import torch

sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/gpic-wt/perf/baselines/PixelGen")


def main(tar, n, out):
    from PIL import Image
    from torchvision import transforms as T
    from src.data.dataset.augmentation import random_crop_arr
    to_t = T.ToTensor(); norm = T.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True)
    st = {k: [] for k in ("read", "json", "decode", "crop", "totensor", "normalize")}
    pixels, caps, tens = [], [], []
    t_all = time.perf_counter()
    with tarfile.open(tar) as tf:
        pending = {}
        for m in tf:
            if not m.isfile():
                continue
            t0 = time.perf_counter(); raw = tf.extractfile(m).read(); t1 = time.perf_counter()
            key, ext = m.name.rsplit(".", 1)
            d = pending.setdefault(key, {}); d[ext] = raw; d.setdefault("_read", 0.0); d["_read"] += t1 - t0
            if "json" in d and len(d) >= 3:  # json + image + _read
                img_ext = [e for e in d if e not in ("json", "_read")][0]
                t0 = time.perf_counter(); meta = json.loads(d["json"]); t1 = time.perf_counter()
                im = Image.open(io.BytesIO(d[img_ext])).convert("RGB"); t2 = time.perf_counter()
                arr = random_crop_arr(im, 256); t3 = time.perf_counter()
                x = to_t(arr); t4 = time.perf_counter()
                norm(x); t5 = time.perf_counter()
                st["read"].append(d["_read"]); st["json"].append(t1 - t0); st["decode"].append(t2 - t1)
                st["crop"].append(t3 - t2); st["totensor"].append(t4 - t3); st["normalize"].append(t5 - t4)
                pixels.append(im.size[0] * im.size[1]); caps.append(meta["caption"]); tens.append(x)
                del pending[key]
                if len(tens) >= n:
                    break
    wall = time.perf_counter() - t_all
    res = {"n": len(tens), "wall_s": wall, "img_per_s_one_proc": len(tens) / wall,
           "per_sample_ms": {k: {"mean": 1e3 * statistics.mean(v), "p50": 1e3 * statistics.median(v),
                                 "p95": 1e3 * float(np.percentile(v, 95))} for k, v in st.items()},
           "src_megapixels": {"mean": statistics.mean(pixels) / 1e6, "p50": statistics.median(pixels) / 1e6,
                              "p95": float(np.percentile(pixels, 95)) / 1e6, "max": max(pixels) / 1e6}}
    # collate and pin, per batch of 32
    b = tens[:32]
    t0 = time.perf_counter()
    for _ in range(20):
        xb = torch.stack(b)
    res["collate_ms_per_batch"] = 1e3 * (time.perf_counter() - t0) / 20
    try:
        t0 = time.perf_counter()
        for _ in range(5):
            xb.pin_memory()
        res["pin_ms_per_batch"] = 1e3 * (time.perf_counter() - t0) / 5
    except Exception as e:
        res["pin_ms_per_batch"] = repr(e)
    # tokenizer, exactly as Qwen3TextEncoder calls it
    from transformers import Qwen2Tokenizer
    tok = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-1.7B", max_length=300, padding_side="right")
    res["tokenizer_class"] = type(tok).__name__
    lens = [len(tok(c).input_ids) for c in caps[:2000]]
    res["caption_tokens"] = {"mean": statistics.mean(lens), "p50": statistics.median(lens),
                             "p95": float(np.percentile(lens, 95)), "max": max(lens), "frac_truncated_at_300": sum(l > 300 for l in lens) / len(lens)}
    tt = []
    for i in range(0, min(len(caps), 640), 32):
        t0 = time.perf_counter()
        tok(caps[i:i + 32], truncation=True, max_length=300, padding="max_length", return_tensors="pt")
        tt.append(time.perf_counter() - t0)
    res["tokenizer_ms_per_batch32"] = {"mean": 1e3 * statistics.mean(tt), "p95": 1e3 * float(np.percentile(tt, 95))}
    # end-to-end webdataset pipeline on one shard, one process
    from src.data.dataset.gpic_dataset import GPICWebDataset
    import os
    ds = GPICWebDataset(tar_dir=os.path.dirname(tar) + "/", tar_pattern=os.path.basename(tar)).create_dataset()
    t0 = time.perf_counter(); k = 0
    for s in ds:
        k += 1
        if k >= n:
            break
    res["webdataset_pipeline_img_per_s_one_proc"] = k / (time.perf_counter() - t0)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
