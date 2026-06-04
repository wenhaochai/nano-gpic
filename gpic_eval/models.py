"""Model constants, selection, and lightweight data containers.

Single source of truth for supported model names and the mapping
between CLI model names and npz key prefixes.
"""

import numpy as np


class FDStatistics:
    """Stores mean and covariance for Frechet Distance computation."""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma


STAT_FIELD_NAMES: dict[str, tuple[str, str]] = {
    "inception": ("inception_mu", "inception_sigma"),
    "dino": ("dino_mu", "dino_sigma"),
}


def stat_field_names(key: str) -> tuple[str, str]:
    """Return (mu_field, sigma_field) for a stats key.

    Canonical keys use the lookup table; other keys use {key}_mu / {key}_sigma.
    """
    key = key.strip()
    if not key:
        raise ValueError("stats key must be a non-empty string")
    return STAT_FIELD_NAMES.get(key, (f"{key}_mu", f"{key}_sigma"))


DINO_HUB_NAME = "dinov2_vitl14"

SUPPORTED_MODELS = ["inception", "dino"]
DEFAULT_MODELS = ["dino"]

ALL_METRICS = ["fd", "prdc", "mmd"]
