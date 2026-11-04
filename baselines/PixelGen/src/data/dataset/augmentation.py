# from https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py
import math
import random
import numpy as np
from PIL import Image


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def random_crop_arr_region(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """random_crop_arr that resizes only the region it keeps: the same random draws in the same order, about 43% less CPU (tools/test_region_crop.py).
    Pillow runs the vertical pass first only if (H - out_h) > 2 (W - out_w) (Resample.c, ImagingResampleInner), and the order changes the 8-bit rounding, so each pass runs alone, in random_crop_arr's order.
    Not bit-identical: Pillow takes the box as single-precision floats. On real images about 0.035% of the channel values differ, by at most 2 of 255."""
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    W, H = pil_image.size
    scale = smaller_dim_size / min(W, H)
    out_w, out_h = round(W * scale), round(H * scale)
    crop_y = random.randrange(out_h - image_size + 1)
    crop_x = random.randrange(out_w - image_size + 1)
    sx, sy = W / out_w, H / out_h
    x0, x1 = crop_x * sx, min(W, (crop_x + image_size) * sx)
    y0, y1 = crop_y * sy, min(H, (crop_y + image_size) * sy)
    if (H - out_h) > 0 and (H - out_h) > (W - out_w) * 2:
        tmp = pil_image.resize((W, image_size), resample=Image.BICUBIC, box=(0, y0, W, y1))
        return np.asarray(tmp.resize((image_size, image_size), resample=Image.BICUBIC, box=(x0, 0, x1, image_size)))
    support = 2 * max(1.0, sy)   # BICUBIC's support, in source rows; the rows outside [r0, r1) have no weight
    r0, r1 = max(0, int(y0 - support) - 1), min(H, int(y1 + support) + 2)
    tmp = pil_image.crop((0, r0, W, r1)).resize((image_size, r1 - r0), resample=Image.BICUBIC, box=(x0, 0, x1, r1 - r0))
    return np.asarray(tmp.resize((image_size, image_size), resample=Image.BICUBIC, box=(0, y0 - r0, image_size, y1 - r0)))


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])

