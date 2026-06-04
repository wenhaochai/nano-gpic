"""
Metrics for image generation evaluation.

Courtesy of https://github.com/layer6ai-labs/dgm-eval/ and https://github.com/clovaai/generative-evaluation-prdc.
"""


from .fd import compute_FD_with_stats, compute_statistics
from .mmd import compute_mmd
from .prdc import compute_prdc
