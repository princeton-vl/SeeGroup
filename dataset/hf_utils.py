import io
import os

import numpy as np


def load_hf_dataset(dataset_name, split, streaming=True, cache_dir=None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            'Hugging Face dataset loading requires the "datasets" package. '
            'Install the release requirements with `pip install -r requirements.txt`.'
        ) from exc

    kwargs = dict(split=split, streaming=streaming)
    if cache_dir is not None:
        kwargs['cache_dir'] = cache_dir
    return load_dataset(dataset_name, **kwargs)


def decode_image(value):
    if isinstance(value, np.ndarray):
        return value

    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        PILImage = ()

    if isinstance(value, PILImage):
        return np.asarray(value.copy())

    if isinstance(value, dict):
        if value.get('bytes') is not None:
            from PIL import Image

            with Image.open(io.BytesIO(value['bytes'])) as image:
                return np.asarray(image.copy())
        if value.get('path') is not None:
            value = value['path']

    if isinstance(value, (str, os.PathLike)):
        from PIL import Image

        with Image.open(value) as image:
            return np.asarray(image.copy())

    if hasattr(value, '__array__'):
        return np.asarray(value)

    raise TypeError(f'Unsupported image value type: {type(value)!r}')


def image_to_float_rgb(value):
    image = decode_image(value)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f'Expected RGB image with 3 dimensions, got shape {image.shape}')
    if image.shape[2] == 4:
        image = image[..., :3]
    if image.shape[2] != 3:
        raise ValueError(f'Expected 3 image channels, got shape {image.shape}')

    image = image.astype(np.float32, copy=False)
    if image.max(initial=0) > 1.0:
        scale = 65535.0 if image.max(initial=0) > 255.0 else 255.0
        image = image / scale
    return image


def depth_png_to_meters(value):
    depth = decode_image(value)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32, copy=False) / 1000.0
    depth[~np.isfinite(depth)] = 0
    depth[depth > 80] = 0
    depth[depth <= 0] = 0
    return depth


def get_row_value(row, names):
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f'None of the expected fields are present: {names}')


def split_hf_stream_by_rank(rows):
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    if world_size <= 1:
        return rows

    try:
        from datasets.distributed import split_dataset_by_node
    except ImportError:
        return (row for idx, row in enumerate(rows) if idx % world_size == rank)
    return split_dataset_by_node(rows, rank=rank, world_size=world_size)
