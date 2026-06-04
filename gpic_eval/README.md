# gpic-eval

Image generation evaluation metrics: Frechet Distance (FD), Precision/Recall/Density/Coverage (PRDC), and MMD over Inception and DINOv2 features. This codebase builds off of [dgm-eval](https://github.com/layer6ai-labs/dgm-eval) and is purposed for evaluation on the Giant Permissive Image Corpus dataset.

For a quick start, see [Basic evaluation](#basic-evaluation).

## Setup

Requires Python 3.10+.

**DINOv2 metrics only** (default -- no TensorFlow needed):

```bash
# after cloning the repository
pip install .
# OR without cloning
pip install "gpic-eval @ git+https://github.com/keshik6/gpic.git#subdirectory=gpic_eval"
```

**All metrics** including TF Inception:

```bash
pip install ".[all]"
# OR
pip install "gpic-eval[all] @ git+https://github.com/keshik6/gpic.git#subdirectory=gpic_eval"
```

The `[all]` extra adds `tensorflow`.

### Conda environment from scratch

```bash
conda create -n gpic-eval python=3.10 -y
conda activate gpic-eval
pip install .  # or pip install ".[all]" to include Inception-v3
```

## Usage

### CLI

The CLI is available as both `gpic-eval` and `python -m gpic_eval`.

#### Basic evaluation

You may find the reference files on [HuggingFace](https://huggingface.co/datasets/stanford-vision-lab/gpic/tree/main/reference_stats). The download links are:
  - [Nano](https://huggingface.co/datasets/stanford-vision-lab/gpic/resolve/main/reference_stats/nano_stats.npz?download=true)
  - [Lite](https://huggingface.co/datasets/stanford-vision-lab/gpic/resolve/main/reference_stats/lite_stats.npz?download=true)
  - [Full](https://huggingface.co/datasets/stanford-vision-lab/gpic/resolve/main/reference_stats/full_stats.npz?download=true)
  - [Validation](https://huggingface.co/datasets/stanford-vision-lab/gpic/resolve/main/reference_stats/val_stats.npz?download=true)
  - [Test](https://huggingface.co/datasets/stanford-vision-lab/gpic/resolve/main/reference_stats/test_stats.npz?download=true)

If `gpic-eval` fails to use an available gpu, please set the `--device` flag.

```bash
# Evaluate samples against precomputed reference stats
gpic-eval sample.npy ref_stats.npz # --device <gpu_id>

# Evaluate samples against a directory of reference images
gpic-eval sample.npy /path/to/reference_images/

# Both samples and references can be directories of images
gpic-eval /path/to/generated_images/ /path/to/reference_images/

# Raw reference images can also be provided as .npy or .npz batches
gpic-eval sample.npy ref_images.npy
gpic-eval sample.npy ref_images.npz
```

#### Output

`gpic-eval` prints progress logs, then a metric table, and also saves the same results as JSON to `out_gpic-eval/<sample_name>_<reference_name>.json` in the current working directory.

The JSON is grouped by model, for example:

```json
{
  "dinov2": {
    "fd": 12.34,
    "fd_mu": 3.21,
    "fd_sigma": 9.13,
    "precision": 0.62,
    "recall": 0.48,
    "density": 0.71,
    "coverage": 0.44,
    "mmd": 0.015
  }
}
```

`fd_mu` and `fd_sigma` are the mean and covariance components of FD. Metrics that cannot be computed from the provided inputs are shown as unavailable in the table and written as `null` in JSON.

#### Input types

**Both samples and references** (first and second positional arguments) accept the same set of formats and are auto-detected. Resolution priority per model is **stats > embeddings > images** for FD, and **embeddings > images** for PRDC/MMD. Pre-computed inputs skip the model forward pass. If both sides are fully pre-computed, the model is never loaded.

Stat keys: `inception_mu`, `inception_sigma`, `dino_mu`, `dino_sigma`.
Embedding keys (optional, used directly for PRDC/MMD): `inception_embeddings`, `dino_embeddings`.
Raw image key: `arr_0`

```bash
# Directory of generated images vs directory of reference images
gpic-eval /path/to/generated_images/ /path/to/reference_images/

# Directory of generated images vs precomputed reference stats
gpic-eval /path/to/generated_images/ ref.npz

# .npy array (NHWC uint8, shape [N, H, W, 3]) vs reference directory
gpic-eval sample.npy /path/to/reference_images/

# .npz with arr_0 key vs .npz with arr_0 key
gpic-eval sample.npz ref_images.npz
```

Note: directories, `.npy`, and `.npz` files containing images all work as references. Stats_embeddings-only `.npz` references are faster, and are what we provide in the precomputed reference files in HuggingFace.

#### Model selection

```bash
# DINOv2 only, default
gpic-eval sample.npy ref.npz # --models dino

# Inception only (requires [all] install)
gpic-eval sample.npy ref.npz --models inception

# Both (requires explicit model selection)
gpic-eval sample.npy ref.npz --models inception,dino
```

#### Metric selection

```bash
# All available metrics (FD, PRDC, MMD)
gpic-eval sample.npy ref.npz

# Only Frechet Distance
gpic-eval sample.npy ref.npz --metrics fd

# FD + Precision/Recall/Density/Coverage
gpic-eval sample.npy ref.npz --metrics fd,prdc

# Multiple metrics
gpic-eval sample.npy ref.npz --metrics fd,prdc,mmd
```

#### Performance tuning

```bash
# Increase batch size and DataLoader workers
gpic-eval sample.npy ref.npz --batch_size 256 --num_workers 8
```

`--batch_size` sizes the DINO loader; the Inception backend runs at
`batch_size // 4`.

## Available metrics


| Metric                                  | `--metrics` flag | Models              | Description                           |
| --------------------------------------- | ---------------- | ------------------- | ------------------------------------- |
| Frechet Distance                        | `fd`             | `dino`, `inception` | FD-DINOv2 or FID                      |
| Precision / Recall / Density / Coverage | `prdc`           | `dino`, `inception` | Manifold-based fidelity and diversity |
| MMD                                     | `mmd`            | `dino`, `inception` | Maximum Mean Discrepancy              |


Metrics that require reference features (PRDC, MMD) need the reference to contain embeddings or raw images. Pre-computed stats alone (mu/sigma) are only sufficient for FD.

# Acknowledgements

This repository builds off the following prior work.

- [dgm-eval](https://github.com/layer6ai-labs/dgm-eval)
- [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations)
- [generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc)
