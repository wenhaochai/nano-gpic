"""TF Inception V3 feature extraction.

All TensorFlow code is isolated in this module.

Adapted from https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py.
"""

import os
import random
from typing import Iterable

import numpy as np
import requests
import tensorflow.compat.v1 as tf
from tqdm.auto import tqdm

INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
_INCEPTION_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "gpic_eval",
)
INCEPTION_V3_PATH = os.path.join(_INCEPTION_CACHE_DIR, "classify_image_graph_def.pb")

FID_POOL_NAME = "pool_3:0"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    def __init__(
        self,
        session,
        batch_size=64,
    ):
        self.sess = session
        self.batch_size = batch_size
        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.pool_features = _create_feature_graph(self.image_input)

    def warmup(self):
        self.compute_activations(np.zeros([1, 8, 64, 64, 3]))

    def read_activations(self, npz_path: str, num_workers: int = 0) -> np.ndarray:
        from torch.utils.data import DataLoader
        from ..utils import make_dataset, collate_uint8

        dataset = make_dataset(npz_path)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers, collate_fn=collate_uint8)
        return self.compute_activations(loader)

    def compute_activations(self, batches: Iterable[np.ndarray]) -> np.ndarray:
        """Compute pool features for downstream evals.

        :param batches: an iterator over NHWC numpy arrays in [0, 255].
        :return: numpy array of shape [N, 2048].
        """
        preds = []
        for batch in tqdm(batches):
            batch = batch.astype(np.float32)
            pred = self.sess.run(
                self.pool_features,
                {self.image_input: batch},
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
        return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Inception model helpers
# ---------------------------------------------------------------------------


def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return
    os.makedirs(_INCEPTION_CACHE_DIR, exist_ok=True)
    print("downloading InceptionV3 model...")
    with requests.get(INCEPTION_V3_URL, stream=True) as r:
        r.raise_for_status()
        tmp_path = INCEPTION_V3_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        os.rename(tmp_path, INCEPTION_V3_PATH)


def _create_feature_graph(input_batch):
    _download_inception_model()
    # Random prefix for TF graph namespace deduplication (needed when
    # multiple Evaluator instances share a graph)
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    (pool3,) = tf.import_graph_def(
        graph_def,
        input_map={f"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME],
        name=prefix,
    )
    _update_shapes(pool3)
    return pool3


def _update_shapes(pool3):
    """Fix batch dimension shapes after importing the frozen graph.

    The frozen Inception graph has a fixed batch dimension of 1.
    This replaces it with None (dynamic) so variable batch sizes work.

    Note: Accesses TF internal APIs (shape._dims, o.__dict__["_shape_val"])
    which may break across TF versions.
    """
    ops = pool3.graph.get_operations()
    for op in ops:
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
                shape = [s for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------


def create_inception_session(batch_size=64, device=None):
    """Create a TF session + Evaluator. Returns (tf, sess, evaluator).

    Args:
        batch_size: Batch size for the evaluator.
        device: Optional device string (e.g. "cuda:1"). If given, pins the
                TF session to the corresponding GPU via CUDA_VISIBLE_DEVICES.
    """
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    if device is not None and device.startswith("cuda:"):
        config.gpu_options.visible_device_list = device.split(":")[1]
    sess = tf.Session(config=config)
    evaluator = Evaluator(sess, batch_size=batch_size)
    evaluator.warmup()
    return tf, sess, evaluator
