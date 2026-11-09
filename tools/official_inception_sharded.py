"""tools/official_inception.py split over GPUs (gpic-28 2026-11-09, the eval idle-GPU work; owner: 「一定要处理空闲」): the one-GPU Inception pass
took 58 s alone and 100-190 s beside DINOv2 and the next predict, and slowed that predict's rank on its GPU. Each shard computes the TF Inception-v3
pool features of a contiguous block of whole 125-image batches (gpic_eval's batch for models=["inception"] at batch_size=500), so every batch,
and with it every feature, is the one the one-GPU pass computes; the metrics step joins the shards in image order and calls the same metric code
(gpic_eval.eval._compute_metrics_from_resolved with the reference's inception stats and embeddings, after np.random.seed(0) as before).
usage: official_inception_sharded.py features <samples .npy|.npz> <shard> <shards> <out.npy>      (one GPU each: CUDA_VISIBLE_DEVICES)
       official_inception_sharded.py metrics <ref_stats.npz> <out.json> <shard_0.npy> ... <shard_S-1.npy>
(env: envs/gpic-eval-tf; XDG_CACHE_HOME holds gpic_eval/classify_image_graph_def.pb)"""
import json, sys, time
import numpy as np

BATCH = 125   # eval_with_ref_stats(batch_size=500) -> _run_inception_pipeline(batch_size=500 // 4)


def features(samples, shard, shards, out):
    t0 = time.time()
    arr = np.load(samples, mmap_mode="r") if samples.endswith(".npy") else np.load(samples)["arr_0"]
    n = arr.shape[0]
    nb = (n + BATCH - 1) // BATCH
    b0, b1 = shard * nb // shards, (shard + 1) * nb // shards
    from gpic_eval.backends.inception import create_inception_session
    tf, sess, ev = create_inception_session(batch_size=BATCH)
    # the one-GPU pass: ImageArrayDataset items stacked by collate_uint8, then astype(float32) in compute_activations
    batches = (np.ascontiguousarray(arr[b * BATCH:min((b + 1) * BATCH, n)]) for b in range(b0, b1))
    f = ev.compute_activations(batches) if b1 > b0 else np.zeros((0, 2048), np.float32)
    np.save(out, f)
    sess.close()
    print(json.dumps({"shard": shard, "images": [b0 * BATCH, min(b1 * BATCH, n)], "features": list(f.shape), "s": round(time.time() - t0, 1)}))


def metrics(ref, out, parts):
    t0 = time.time()
    from gpic_eval.eval import ALL_METRICS, detect_batch_format, _resolve_inputs, _compute_metrics_from_resolved
    np.random.seed(0)   # as tools/official_inception.py (imports, then the seed): PRDC draws its 10K subsample with np.random.choice
    feats = np.concatenate([np.load(p) for p in parts], axis=0)
    ref_batch = detect_batch_format(ref)
    ref_stats, ref_emb, ref_extract, ref_src = _resolve_inputs(ref_batch, "inception", "reference", True)
    assert not ref_extract, f"the reference {ref} has no inception stats and embeddings"
    m = _compute_metrics_from_resolved(None, feats, ref_stats, ref_emb, list(ALL_METRICS), "inception",
                                       f"{len(parts)} shards (shape={feats.shape})", ref_src)
    res = {f"inception_{k}": float(m[k]) for k in ("fd", "fd_mu", "fd_sigma", "precision", "recall", "density", "coverage", "mmd") if k in m}
    res["inception_s"] = round(time.time() - t0, 1)
    res["inception_shards"] = len(parts)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "features":
        features(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    elif mode == "metrics":
        metrics(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        sys.exit(__doc__)
