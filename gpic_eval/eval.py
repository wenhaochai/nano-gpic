"""Core evaluation orchestrator (programmatic API).

Provides the following use cases:
  1. eval_with_ref_stats()    — full eval against precomputed reference stats
  2. eval_from_embeddings()   — FD/PRDC/MMD from custom embedding arrays
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .models import ALL_METRICS, DEFAULT_MODELS, FDStatistics, SUPPORTED_MODELS
from .metrics import compute_FD_with_stats, compute_prdc, compute_mmd
from torch.utils.data import DataLoader

from .utils import fd_stats_from_features, make_dataset

from ._ref_format import (  # noqa: F401
    BatchInfo,
    detect_batch_format,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Metric results from evaluation."""

    fd: dict[str, float] = field(default_factory=dict)
    fd_mu: dict[str, float] = field(default_factory=dict)
    fd_sigma: dict[str, float] = field(default_factory=dict)
    precision: dict[str, float] = field(default_factory=dict)
    recall: dict[str, float] = field(default_factory=dict)
    density: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    mmd: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_MAX_PRDC_SAMPLES = 10_000
_RECOMMENDED_SAMPLES = 50_000


def _warn_low_samples(metric_name, n_ref, n_sample, recommended=_RECOMMENDED_SAMPLES):
    if n_ref < recommended or n_sample < recommended:
        warnings.warn(
            f"{metric_name}: using {n_ref} ref and {n_sample} sample features; "
            f"{recommended} recommended for reliable results"
        )


def _compute_metrics_from_resolved(
    sample_stats, sample_feats, ref_stats, ref_feats,
    selected_metrics, model_key, sample_source, ref_source,
):
    """Compute selected metrics given resolved stats/features for both sides.

    Each side may provide stats only, features (from which stats are derived),
    or both. Prints a clear message for any metric that is skipped.
    """
    results = {}

    # ------------------------------------------------------------------
    # FD — uses (mu, sigma) for both sides. Stats > derive from features.
    # ------------------------------------------------------------------
    if "fd" in selected_metrics:
        s_stats = sample_stats if sample_stats is not None else (
            fd_stats_from_features(sample_feats) if sample_feats is not None else None)
        r_stats = ref_stats if ref_stats is not None else (
            fd_stats_from_features(ref_feats) if ref_feats is not None else None)
        if s_stats is not None and r_stats is not None:
            fd_total, fd_mu, fd_sigma = compute_FD_with_stats(
                s_stats.mu, r_stats.mu, s_stats.sigma, r_stats.sigma)
            results["fd"] = float(fd_total)
            results["fd_mu"] = float(fd_mu)
            results["fd_sigma"] = float(fd_sigma)

    # ------------------------------------------------------------------
    # MMD and PRDC — require feature arrays for both sides.
    # ------------------------------------------------------------------
    if "mmd" in selected_metrics:
        if sample_feats is None or ref_feats is None:
            missing = "sample" if sample_feats is None else "reference"
            src = sample_source if missing == "sample" else ref_source
            print(f"[{model_key}] MMD skipped — {missing} has no features "
                  f"(source: {src})")
        else:
            _warn_low_samples("MMD", len(ref_feats), len(sample_feats))
            mmds = compute_mmd(ref_feats, sample_feats)
            results["mmd"] = float(np.mean(mmds))

    if "prdc" in selected_metrics:
        if sample_feats is None or ref_feats is None:
            missing = "sample" if sample_feats is None else "reference"
            src = sample_source if missing == "sample" else ref_source
            print(f"[{model_key}] PRDC skipped — {missing} has no features "
                  f"(source: {src})")
        else:
            reduced_n = min(_MAX_PRDC_SAMPLES, len(ref_feats), len(sample_feats))
            ref_sub = ref_feats
            sample_sub = sample_feats
            if len(ref_feats) > reduced_n:
                ref_sub = ref_feats[np.random.choice(len(ref_feats), reduced_n, replace=False)]
            if len(sample_feats) > reduced_n:
                sample_sub = sample_feats[np.random.choice(len(sample_feats), reduced_n, replace=False)]
            prdc = compute_prdc(ref_sub, sample_sub, nearest_k=5)
            results["precision"] = float(prdc['precision'])
            results["recall"] = float(prdc['recall'])
            results["density"] = float(prdc['density'])
            results["coverage"] = float(prdc['coverage'])

    return results


def _store_metrics(result, model_key, metrics):
    """Write metric dict into EvalResult."""
    _FIELD_MAP = {
        "fd": "fd", "fd_mu": "fd_mu", "fd_sigma": "fd_sigma",
        "precision": "precision", "recall": "recall",
        "density": "density", "coverage": "coverage",
        "mmd": "mmd",
    }
    for metric_name, field_name in _FIELD_MAP.items():
        if metric_name in metrics:
            getattr(result, field_name)[model_key] = metrics[metric_name]


def _resolve_inputs(batch: BatchInfo, model_key: str, side: str, need_features: bool):
    """Resolve (stats, embeddings, needs_extraction, source_label) for one batch.

    FD prefers stats > embeddings > raw images.
    PRDC/MMD prefer embeddings > raw images (stats cannot satisfy them).

    Returns:
        stats: FDStatistics or None (precomputed stats if present, else None;
            FD will derive from features when stats are None).
        embeddings: np.ndarray or None (precomputed features array, if any).
        needs_extraction: True iff raw images must be processed (no stats and no
            embeddings, or PRDC/MMD requested but no embeddings).
        source_label: human-readable description of what will be used.
    """
    s = batch.stats.get(model_key)
    emb = batch.embeddings.get(model_key)

    parts = []
    if s is not None:
        parts.append(f"stats(mu shape={s.mu.shape})")
    if emb is not None:
        parts.append(f"embeddings(shape={emb.shape})")

    extract = False
    if s is None and emb is None:
        # Need everything from images.
        if batch.is_image_batch:
            extract = True
            label = f"raw images at {batch.path}"
        else:
            label = f"no {model_key} data available in {batch.path}"
    elif need_features and emb is None:
        # Have stats (for FD) but PRDC/MMD need features → extract from images.
        if batch.is_image_batch:
            extract = True
            parts.append("raw images for PRDC/MMD")
        label = f"{' + '.join(parts)} from {batch.path}"
    else:
        label = f"{' + '.join(parts)} from {batch.path}"

    return s, emb, extract, label


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def _run_dino_pipeline(sample_batch: BatchInfo, ref_batch: BatchInfo,
                      batch_size, device, result,
                      selected_metrics=None, num_workers=0):
    """Run DINOv2 evaluation pipeline, writing results into *result*."""
    if selected_metrics is None:
        selected_metrics = list(ALL_METRICS)
    need_features = "prdc" in selected_metrics or "mmd" in selected_metrics

    sample_stats, sample_emb, sample_extract, sample_src = _resolve_inputs(
        sample_batch, "dino", "sample", need_features)
    ref_stats, ref_emb, ref_extract, ref_src = _resolve_inputs(
        ref_batch, "dino", "reference", need_features)

    print(f"[dino] reference: {ref_src}")
    print(f"[dino] sample: {sample_src}")

    sample_feats = sample_emb
    ref_feats = ref_emb

    if sample_extract or ref_extract:
        from .backends.dino import build_dinov2, extract_features_from_npz, _dino_transform

        print(f"[dino] loading DINOv2 on {device}...")
        model = build_dinov2(device)

        if ref_extract:
            print("[dino] extracting features for reference batch...")
            ref_dataset = make_dataset(ref_batch.path, transform=_dino_transform)
            ref_loader = DataLoader(ref_dataset, batch_size=batch_size, num_workers=num_workers)
            ref_feats = extract_features_from_npz(ref_loader, model, device)

        if sample_extract:
            print("[dino] extracting features for sample batch...")
            sample_dataset = make_dataset(sample_batch.path, transform=_dino_transform)
            sample_loader = DataLoader(sample_dataset, batch_size=batch_size, num_workers=num_workers)
            sample_feats = extract_features_from_npz(sample_loader, model, device)
    else:
        print("[dino] skipping model load — both sides have pre-computed inputs")

    print("[dino] computing metrics...")
    metrics = _compute_metrics_from_resolved(
        sample_stats, sample_feats, ref_stats, ref_feats,
        selected_metrics, "dino", sample_src, ref_src,
    )
    _store_metrics(result, "dino", metrics)


def _run_inception_pipeline(sample_batch: BatchInfo, ref_batch: BatchInfo,
                            batch_size, result,
                            selected_metrics=None, num_workers=0,
                            device=None):
    """Run Inception evaluation pipeline, writing results into *result*."""
    if selected_metrics is None:
        selected_metrics = list(ALL_METRICS)
    need_features = "prdc" in selected_metrics or "mmd" in selected_metrics

    sample_stats, sample_emb, sample_extract, sample_src = _resolve_inputs(
        sample_batch, "inception", "sample", need_features)
    ref_stats, ref_emb, ref_extract, ref_src = _resolve_inputs(
        ref_batch, "inception", "reference", need_features)

    print(f"[inception] reference: {ref_src}")
    print(f"[inception] sample: {sample_src}")

    sample_feats = sample_emb
    ref_feats = ref_emb
    tf = sess = None

    if sample_extract or ref_extract:
        from .backends.inception import create_inception_session

        print("[inception] importing TensorFlow ...")
        tf, sess, evaluator = create_inception_session(batch_size=batch_size, device=device)

        if ref_extract:
            print("[inception] extracting activations for reference batch...")
            ref_feats = evaluator.read_activations(ref_batch.path, num_workers=num_workers)

        if sample_extract:
            print("[inception] extracting activations for sample batch...")
            sample_feats = evaluator.read_activations(sample_batch.path, num_workers=num_workers)
    else:
        print("[inception] skipping model load — both sides have pre-computed inputs")

    print("[inception] computing metrics...")
    metrics = _compute_metrics_from_resolved(
        sample_stats, sample_feats, ref_stats, ref_feats,
        selected_metrics, "inception", sample_src, ref_src,
    )
    _store_metrics(result, "inception", metrics)

    if sess is not None:
        sess.close()
        tf.reset_default_graph()


# ---------------------------------------------------------------------------
# Use Case 1: Eval with precomputed reference stats
# ---------------------------------------------------------------------------


def eval_with_ref_stats(
    sample_path: str,
    ref_path: str,
    models: Optional[list[str]] = None,
    metrics: Optional[list[str]] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    device: Optional[str] = None,
) -> EvalResult:
    """Full evaluation against precomputed reference stats.

    Args:
        sample_path: Path to .npy/.npz with sample images, or image directory.
            May also be an .npz with pre-computed `*_mu`/`*_sigma` and/or
            `*_embeddings` keys.
        ref_path: Path to reference batch or pre-computed stats.
        models: List of model names (default: ["dino"]).
        metrics: List of metric names (default: all).
        batch_size: Batch size for feature extraction.
        num_workers: Number of DataLoader workers.

    Returns:
        EvalResult with all computed metrics.
    """
    selected = list(models) if models else list(DEFAULT_MODELS)
    invalid = [m for m in selected if m not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError(f"unknown model(s): {invalid}. Available: {SUPPORTED_MODELS}")
    selected_metrics = list(metrics) if metrics else list(ALL_METRICS)

    sample_batch = detect_batch_format(sample_path)
    ref_batch = detect_batch_format(ref_path)
    print(f"sample: {sample_batch.path} (image_batch={sample_batch.is_image_batch}, "
          f"stats={sorted(sample_batch.stats.keys())}, "
          f"embeddings={sorted(sample_batch.embeddings.keys())})")
    print(f"reference: {ref_batch.path} (image_batch={ref_batch.is_image_batch}, "
          f"stats={sorted(ref_batch.stats.keys())}, "
          f"embeddings={sorted(ref_batch.embeddings.keys())})")

    result = EvalResult()

    # ------------------------------------------------------------------
    # DINOv2 pipeline
    # ------------------------------------------------------------------
    if "dino" in selected:
        import torch
        torch_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch_device.type == "cpu":
            warnings.warn(
                "DINOv2 falling back to CPU — no GPU detected. "
                "Pass --device cuda:0 (or another cuda index) if a GPU is available.",
                stacklevel=2,
            )
        _run_dino_pipeline(sample_batch, ref_batch, batch_size, torch_device, result,
                           selected_metrics=selected_metrics,
                           num_workers=num_workers)

    # ------------------------------------------------------------------
    # Inception pipeline
    # ------------------------------------------------------------------
    if "inception" in selected:
        # Inception pool features are 2048-d (~2x DINO's 1024-d), and TF's
        # graph runs preprocessing on-device, so it has roughly 4x the
        # activation memory of DINO at the same batch size. Quarter it.
        inception_batch_size = max(1, batch_size // 4)
        _run_inception_pipeline(sample_batch, ref_batch, inception_batch_size, result,
                                selected_metrics=selected_metrics,
                                num_workers=num_workers,
                                device=device)

    return result


# ---------------------------------------------------------------------------
# Use Case 2: Eval from custom embeddings
# ---------------------------------------------------------------------------


def eval_from_embeddings(
    sample_embeddings: dict[str, np.ndarray],
    ref_embeddings: Optional[dict[str, np.ndarray]] = None,
    ref_stats: Optional[dict[str, FDStatistics]] = None,
    metrics: Optional[list[str]] = None,
) -> EvalResult:
    """Compute metrics from pre-extracted embeddings.

    For each model key present in sample_embeddings, computes all
    selected metrics using available reference data.

    Args:
        sample_embeddings: {model_key: [N, D] array}.
        ref_embeddings: {model_key: [N, D] array} for metrics needing ref features.
        ref_stats: {model_key: FDStatistics} for FD computation.
        metrics: List of metric names (default: all).

    Returns:
        EvalResult with computed metrics.
    """
    if ref_embeddings is None:
        ref_embeddings = {}
    if ref_stats is None:
        ref_stats = {}
    selected_metrics = list(metrics) if metrics else list(ALL_METRICS)

    result = EvalResult()

    for key, sample_feats in sample_embeddings.items():
        ref_s = ref_stats.get(key)
        ref_f = ref_embeddings.get(key)
        m = _compute_metrics_from_resolved(
            None, sample_feats, ref_s, ref_f,
            selected_metrics, key,
            f"in-memory embeddings (shape={sample_feats.shape})",
            f"in-memory {'stats' if ref_s is not None else 'embeddings' if ref_f is not None else 'none'}",
        )
        _store_metrics(result, key, m)

    return result
