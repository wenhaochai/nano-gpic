"""Batch format detection (used for both reference and sample inputs).

Detects whether a path contains raw images, pre-computed embeddings, or
pre-computed statistics, and loads any available mu/sigma pairs and
``*_embeddings`` arrays into a ``BatchInfo`` dataclass.
"""

import os
from dataclasses import dataclass, field

import numpy as np

from .models import FDStatistics


_EMBEDDINGS_SUFFIX = "_embeddings"


@dataclass
class BatchInfo:
    """Describes what was detected in a batch (sample or reference) argument."""

    is_image_batch: bool
    path: str
    stats: dict[str, FDStatistics] = field(default_factory=dict)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)

def detect_batch_format(path: str) -> BatchInfo:
    """Auto-detect the contents of a batch path (sample or reference).

    Detection logic:
    1. directory or ``.npy`` -> raw image batch
    2. ``.npz`` with ``arr_0`` -> raw image batch
    3. ``.npz`` with ``*_embeddings`` keys -> pre-computed embeddings
       (still allows PRDC/MMD without raw images)
    4. ``.npz`` with named ``*_mu`` / ``*_sigma`` keys -> FD stats only
    """
    if os.path.isdir(path):
        return BatchInfo(is_image_batch=True, path=path)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        return BatchInfo(is_image_batch=True, path=path)

    if ext == ".npz":
        obj = np.load(path)
        keys = set(obj.keys())
        has_images = "arr_0" in keys

        # Validate that every *_mu key has a matching *_sigma
        for k in keys:
            if k.endswith("_mu"):
                sigma_k = k[:-3] + "_sigma"
                if sigma_k not in keys:
                    raise ValueError(
                        f"incomplete stats in {path}: found {k!r} "
                        f"but missing {sigma_k!r}"
                    )

        stats = {
            k[:-3]: FDStatistics(obj[k], obj[k[:-3] + "_sigma"])
            for k in keys
            if k.endswith("_mu")
        }
        embeddings = {
            k[: -len(_EMBEDDINGS_SUFFIX)]: obj[k]
            for k in keys
            if k.endswith(_EMBEDDINGS_SUFFIX)
        }
        if stats or has_images or embeddings:
            return BatchInfo(
                is_image_batch=has_images,
                path=path,
                stats=stats,
                embeddings=embeddings,
            )

    raise ValueError(
        f"cannot detect batch format for {path} "
        f"(extension={ext!r}). Expected .npy, .npz, or image directory"
    )

