"""Unified CLI for image generation evaluation.

Both sample and reference paths accept the same set of formats and are
auto-detected. Resolution priority per model is: stats > embeddings > images
for FD; embeddings > images for PRDC/MMD.

Usage:
    # Full eval with raw images on both sides:
    python -m gpic_eval sample.npy ref.npy

    # Pre-computed reference stats/embeddings (auto-detected):
    python -m gpic_eval sample.npy ref_stats.npz

    # Pre-computed sample stats/embeddings (auto-detected):
    python -m gpic_eval sample_stats.npz ref_stats.npz
"""

import argparse
import json
import os

from .eval import EvalResult, eval_with_ref_stats
from .models import ALL_METRICS, DEFAULT_MODELS, SUPPORTED_MODELS

_NA = "N/A (stats not available)"


def _parse_csv(arg: str, allowed: list[str], flag_name: str) -> list[str]:
    """Parse a comma-separated CLI flag into a deduplicated list."""
    selected: list[str] = []
    seen: set[str] = set()
    for item in arg.split(","):
        item = item.strip().lower()
        if item and item not in seen:
            seen.add(item)
            selected.append(item)
    if not selected:
        raise ValueError(f"{flag_name} must specify at least one value")
    for item in selected:
        if item not in allowed:
            raise ValueError(
                f"unknown {flag_name} value: {item}. Available: {', '.join(allowed)}"
            )
    return selected


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_table(groups: list[tuple[str, list[tuple[str, str]]]]):
    """Print a Unicode box-drawing table grouped by model.

    *groups* is a list of ``(model_name, [(metric, value), ...])``.
    """
    all_rows = [(m, v) for _, rows in groups for m, v in rows]
    if not all_rows:
        return
    col1_width = max(len(r[0]) for r in all_rows) + 2
    col2_width = max(len(r[1]) for r in all_rows) + 2
    # Ensure header labels fit
    col1_width = max(col1_width, len("Metric") + 2)
    col2_width = max(col2_width, len("Value") + 2)

    h1 = "─" * (col1_width + 2)
    h2 = "─" * (col2_width + 2)
    top = f"┌{h1}┬{h2}┐"
    mid = f"├{h1}┼{h2}┤"
    bot = f"└{h1}┴{h2}┘"
    full = lambda label: f"│ {label:<{col1_width + col2_width + 3}} │"

    def row_str(c1, c2):
        return f"│ {c1:<{col1_width}} │ {c2:<{col2_width}} │"

    print(top)
    print(row_str("Metric", "Value"))
    for model_name, rows in groups:
        print(mid)
        print(full(model_name))
        print(mid)
        for name, value in rows:
            print(row_str(name, value))
    print(bot)


def _build_json(result: EvalResult, run_inception: bool, run_dino: bool,
                selected_metrics: list[str]) -> dict:
    """Build a nested {model: {metric: value}} dict for JSON output."""
    def val(v):
        return float(v) if v is not None else None

    out = {}
    if run_inception:
        inc = {}
        if "fd" in selected_metrics:
            inc["fd"] = val(result.fd.get("inception"))
            inc["fd_mu"] = val(result.fd_mu.get("inception"))
            inc["fd_sigma"] = val(result.fd_sigma.get("inception"))
        if "prdc" in selected_metrics:
            inc["precision"] = val(result.precision.get("inception"))
            inc["recall"] = val(result.recall.get("inception"))
            inc["density"] = val(result.density.get("inception"))
            inc["coverage"] = val(result.coverage.get("inception"))
        if "mmd" in selected_metrics:
            inc["mmd"] = val(result.mmd.get("inception"))
        out["inception"] = inc
    if run_dino:
        dino = {}
        if "fd" in selected_metrics:
            dino["fd"] = val(result.fd.get("dino"))
            dino["fd_mu"] = val(result.fd_mu.get("dino"))
            dino["fd_sigma"] = val(result.fd_sigma.get("dino"))
        if "prdc" in selected_metrics:
            dino["precision"] = val(result.precision.get("dino"))
            dino["recall"] = val(result.recall.get("dino"))
            dino["density"] = val(result.density.get("dino"))
            dino["coverage"] = val(result.coverage.get("dino"))
        if "mmd" in selected_metrics:
            dino["mmd"] = val(result.mmd.get("dino"))
        out["dinov2"] = dino
    return out


def _format_results(result: EvalResult, run_inception: bool, run_dino: bool,
                     selected_metrics: list[str]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Format an EvalResult into groups of ``(model_name, [(metric, value)])``."""
    def fmt(val, fmt_str=".3f", na=_NA):
        return f"{val:{fmt_str}}" if val is not None else na

    groups = []
    if run_inception:
        rows = []
        if "fd" in selected_metrics:
            rows.append(("FD", fmt(result.fd.get("inception"))))
            rows.append(("FD (mu component)", fmt(result.fd_mu.get("inception"))))
            rows.append(("FD (sigma component)", fmt(result.fd_sigma.get("inception"))))
        if "prdc" in selected_metrics:
            rows.append(("Precision", fmt(result.precision.get("inception"))))
            rows.append(("Recall", fmt(result.recall.get("inception"))))
            rows.append(("Density", fmt(result.density.get("inception"))))
            rows.append(("Coverage", fmt(result.coverage.get("inception"))))
        if "mmd" in selected_metrics:
            rows.append(("MMD", fmt(result.mmd.get("inception"))))
        groups.append(("Inception", rows))
    if run_dino:
        rows = []
        if "fd" in selected_metrics:
            rows.append(("FD", fmt(result.fd.get("dino"))))
            rows.append(("FD (mu component)", fmt(result.fd_mu.get("dino"))))
            rows.append(("FD (sigma component)", fmt(result.fd_sigma.get("dino"))))
        if "prdc" in selected_metrics:
            rows.append(("Precision", fmt(result.precision.get("dino"))))
            rows.append(("Recall", fmt(result.recall.get("dino"))))
            rows.append(("Density", fmt(result.density.get("dino"))))
            rows.append(("Coverage", fmt(result.coverage.get("dino"))))
        if "mmd" in selected_metrics:
            rows.append(("MMD", fmt(result.mmd.get("dino"))))
        groups.append(("DINOv2", rows))
    return groups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate image generation metrics"
    )
    parser.add_argument(
        "sample_batch",
        help="path to sample batch: directory of images, .npy with images, "
             ".npz with arr_0/`*_embeddings`/`*_mu`+`*_sigma` (auto-detected).",
    )
    parser.add_argument(
        "ref_batch",
        help="path to reference batch: directory of images, .npy with images, "
             ".npz with arr_0/`*_embeddings`/`*_mu`+`*_sigma` (auto-detected).",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader workers (default: 0)")
    parser.add_argument(
        "--models", type=str, default="dino",
        help="Comma-separated model pipelines to run. Options: inception, dino. "
             "Default: dino",
    )
    parser.add_argument(
        "--metrics", type=str, default=None,
        help="Comma-separated metrics to compute. "
             f"Options: {', '.join(ALL_METRICS)}. Default: all.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run models on (e.g. cuda:0, cuda:1). Default: auto-detect.",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    selected = _parse_csv(args.models, SUPPORTED_MODELS, "--models")
    selected_metrics = (_parse_csv(args.metrics, ALL_METRICS, "--metrics")
                        if args.metrics else list(ALL_METRICS))

    result = eval_with_ref_stats(
        args.sample_batch, args.ref_batch,
        models=selected,
        metrics=selected_metrics,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    groups = _format_results(result, "inception" in selected, "dino" in selected,
                             selected_metrics)
    if groups:
        print()
        _print_table(groups)

    # Save results to JSON
    out_dir = os.path.join(os.getcwd(), "out_gpic-eval")
    os.makedirs(out_dir, exist_ok=True)
    sample_name = os.path.splitext(os.path.basename(args.sample_batch))[0]
    reference_name = os.path.splitext(os.path.basename(args.ref_batch))[0]
    out_path = os.path.join(out_dir, f"{sample_name}_{reference_name}.json")
    metrics_dict = _build_json(result, "inception" in selected, "dino" in selected,
                               selected_metrics)
    with open(out_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
