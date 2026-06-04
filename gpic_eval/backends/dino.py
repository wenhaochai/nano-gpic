"""PyTorch DINOv2 feature extraction (dinov2_vitl14).

All PyTorch DINOv2 code is isolated in this module.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

from ..models import DINO_HUB_NAME

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_dinov2(
    device: "torch.device",
    device_ids: "list[int] | None" = None,
) -> "torch.nn.Module":
    """Load DINOv2 ViT-L/14 via torch.hub.

    If *device_ids* has multiple entries, wraps the model in ``DataParallel``.
    """
    import torch  # noqa: F811

    model = torch.hub.load("facebookresearch/dinov2", DINO_HUB_NAME)
    model.eval().to(device)
    if device_ids and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


def _dino_transform(arr):
    """Per-image transform: HWC uint8 numpy -> [3, 224, 224] float tensor."""
    import torchvision.transforms as TF
    from PIL import Image

    return TF.Compose([
        TF.Resize((224, 224), TF.InterpolationMode.BICUBIC),
        TF.ToTensor(),
        TF.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])(Image.fromarray(arr))


def preprocess_numpy_batch_dino(
    batch: np.ndarray, device: "torch.device"
) -> "torch.Tensor":
    """Convert NHWC uint8 numpy batch to DINOv2-ready tensor.

    Used by callers that have an in-memory numpy batch (e.g., chunked/streaming
    reference paths fed by WebDataset). The eval path performs this work in the
    DataLoader workers via ``_dino_transform`` and does not call this.
    """
    import torch  # noqa: F811

    if isinstance(batch, torch.Tensor):
        batch = batch.numpy()
    tensors = [_dino_transform(img) for img in batch]
    return torch.stack(tensors).to(device, non_blocking=True)


def make_dino_forward_fn(
    model: "torch.nn.Module", device: "torch.device"
):
    """Return a callable that extracts DINOv2 features from a batch.

    Accepts either a preprocessed [B, 3, 224, 224] float tensor (e.g. from a
    DataLoader whose Dataset applied ``_dino_transform``) or an NHWC uint8
    numpy batch. Returns [B, D] float64 features.
    """

    import torch  # noqa: F811

    @torch.no_grad()
    def forward(batch):
        if isinstance(batch, torch.Tensor):
            x = batch.to(device, non_blocking=True)
        else:
            x = preprocess_numpy_batch_dino(batch, device)
        features = model(x)
        return features.cpu().numpy().astype(np.float64)

    return forward


def extract_features_from_npz(
    batches: "Iterable[torch.Tensor]",
    model: "torch.nn.Module",
    device: "torch.device",
) -> np.ndarray:
    """Extract DINOv2 features from preprocessed tensor batches.

    Args:
        batches: iterable of [B, 3, 224, 224] float tensors (already resized,
            ImageNet-normalized — see ``_dino_transform``).
        model: DINOv2 model (any variant).
        device: torch device.

    Returns:
        [N, D] numpy array of float32 features.  Intentionally float32
        (not float64 like ``make_dino_forward_fn``) because this is used
        for P&R manifold distance computation, which operates in float32.
    """
    import torch  # noqa: F811

    all_features = []
    with torch.no_grad():
        for batch in tqdm(batches, desc="DINOv2 features"):
            x = batch.to(device, non_blocking=True)
            features = model(x)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)
