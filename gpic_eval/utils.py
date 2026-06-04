"""Shared utilities: image datasets, array helpers, image transforms, and file management."""

import io
import os
import tempfile
import zipfile

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .models import FDStatistics, stat_field_names as _stats_field_names


# ---------------------------------------------------------------------------
# Image dataset readers
# ---------------------------------------------------------------------------


class ImageArrayDataset(Dataset):
    """Indexes into an in-memory NHWC uint8 array loaded from .npy/.npz."""

    def __init__(self, path, transform=None):
        self.arr = np.load(path) if path.endswith(".npy") else np.load(path)["arr_0"]
        self.transform = transform

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        x = self.arr[idx]
        return self.transform(x) if self.transform is not None else x


class ImagePathDataset(Dataset):
    """Lazily loads images from a directory of image files."""

    _EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root_dir, followlinks=False, transform=None):
        self.paths = sorted(
            os.path.join(dirpath, fname)
            for dirpath, _, fnames in os.walk(root_dir, followlinks=followlinks)
            for fname in fnames
            if os.path.splitext(fname)[1].lower() in self._EXTENSIONS
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.array(Image.open(self.paths[idx]).convert("RGB"), dtype=np.uint8)
        return self.transform(x) if self.transform is not None else x


def make_dataset(path: str, transform=None) -> Dataset:
    """Return an ImageArrayDataset for files or ImagePathDataset for directories."""
    if os.path.isdir(path):
        return ImagePathDataset(path, transform=transform)
    return ImageArrayDataset(path, transform=transform)


# ---------------------------------------------------------------------------
# NPZ key inspection and incremental write
# ---------------------------------------------------------------------------


def get_npz_keys(path: str) -> set[str]:
    """Return the set of key names (without .npy suffix) in an npz file."""
    if not os.path.isfile(path):
        return set()
    with zipfile.ZipFile(path, "r") as zf:
        return {name.removesuffix(".npy") for name in zf.namelist()}


def append_array_to_npz(npz_path: str, key: str, arr: np.ndarray) -> None:
    """Append a numpy array to an existing npz file via zipfile append."""
    npy_key = f"{key}.npy"
    with zipfile.ZipFile(npz_path, "a") as zf:
        if npy_key in zf.namelist():
            print(f"  key {key} already exists, skipping append")
            return
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        zf.writestr(npy_key, buf.read())
    print(f"  appended {key}: shape={arr.shape}, dtype={arr.dtype}")


def save_or_append_npz(
    path: str, arrays_dict: dict[str, np.ndarray], overwrite_keys: bool = False
) -> list[str]:
    """Create an npz or append missing keys to an existing one.

    Args:
        path: Output .npz file path.
        arrays_dict: Mapping of key -> numpy array to save.
        overwrite_keys: If True, overwrite existing keys. If False (default),
            skip keys that already exist.

    Returns:
        List of keys that were actually written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not os.path.isfile(path):
        np.savez(path, **arrays_dict)
        return list(arrays_dict.keys())

    existing = get_npz_keys(path)
    written = []
    for key, arr in arrays_dict.items():
        if key in existing and not overwrite_keys:
            continue
        if key in existing:
            # Must rebuild the file to overwrite
            strip_npz_keys(path, [key])
        append_array_to_npz(path, key, arr)
        written.append(key)
    return written


def strip_npz_keys(path: str, keys_to_remove: list[str]) -> None:
    """Remove specific keys from an npz file.

    Creates a new zip without those entries, then atomically replaces the original.
    """
    npy_keys = {f"{k}.npy" for k in keys_to_remove}

    with zipfile.ZipFile(path, "r") as zin:
        existing = set(zin.namelist())
        to_remove = npy_keys & existing
        if not to_remove:
            return

        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".npz", dir=os.path.dirname(path) or "."
        )
        os.close(tmp_fd)

        try:
            with zipfile.ZipFile(tmp_path, "w") as zout:
                for item in zin.infolist():
                    if item.filename not in npy_keys:
                        data = zin.read(item.filename)
                        zout.writestr(item, data)
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------


_PIL_RESAMPLE = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}


def resize(x, size, mode="bicubic"):
    """Resize to ``[size, size]`` using Pillow.

    Accepts a PIL Image or an HWC uint8 numpy array.  Returns the same type.

    Args:
        x: PIL Image or HWC uint8 ndarray.
        size: Target spatial dimension (square).
        mode: Interpolation mode (``'bicubic'`` or ``'bilinear'``).
    """
    resample = _PIL_RESAMPLE[mode]

    if isinstance(x, Image.Image):
        return x.resize((size, size), resample)

    pil_img = Image.fromarray(x).resize((size, size), resample)
    return np.array(pil_img, dtype=x.dtype)


def resize_center_crop(img, size=256):
    """Center-crop to a square, then resize to size x size, return uint8 array.

    Args:
        img: PIL Image or numpy ndarray (HWC uint8).
        size: Target square dimension.

    Returns:
        numpy array of shape [size, size, 3], dtype uint8.
    """
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")

    w, h = img.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))

    return np.array(resize(img, size), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def fd_stats_from_features(features: np.ndarray) -> FDStatistics:
    """Compute FDStatistics (mean + covariance) from a feature matrix."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return FDStatistics(mu, sigma)


def collate_uint8(batch):
    """Stack a list of uint8 numpy arrays into a single batch array."""
    return np.stack(batch, axis=0)


# ---------------------------------------------------------------------------
# Statistics I/O
# ---------------------------------------------------------------------------


def save_statistics(
    path: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    key: str | None = None,
) -> None:
    """Save precomputed statistics to an .npz file with explicit model identity."""
    if key is None:
        raise ValueError(
            "save_statistics() requires key='...' so the saved file keeps model identity"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mu_key, sigma_key = _stats_field_names(key)
    np.savez(path, **{mu_key: mu, sigma_key: sigma})
    print(f"Saved statistics to {path}")
