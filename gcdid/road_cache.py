"""Read image-derived road labels and share the CUDA mask between both methods."""
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from .ground.settings import ROAD
from .sampling import nearest_axis
from .temporal import ROAD_FORMAT


def read_masks(path):
    """Read only the road planes from the compressed input when storage permits."""
    with np.load(path, allow_pickle=True) as source:
        keep = [i for i, label in enumerate(source['labels']) if str(label).lower() in ROAD]
        native_shape = (int(source['image_height']), int(source['image_width']))
        if not keep:
            return torch.empty((0, *native_shape), dtype=torch.uint8)
        with ZipFile(path) as archive, archive.open('masks.npy') as stream:
            version = np.lib.format.read_magic(stream)
            readers = {(1, 0): np.lib.format.read_array_header_1_0,
                       (2, 0): np.lib.format.read_array_header_2_0}
            if version not in readers:
                return torch.from_numpy(np.ascontiguousarray(source['masks'][keep]))
            shape, fortran, dtype = readers[version](stream)
            if fortran or len(shape) != 3 or dtype.kind not in ('u', 'b'):
                return torch.from_numpy(np.ascontiguousarray(source['masks'][keep]))
            plane = shape[1] * shape[2] * dtype.itemsize
            position = stream.tell()
            selected = []
            for index in keep:
                stream.seek(position + index * plane)
                selected.append(np.frombuffer(stream.read(plane), dtype=dtype).reshape(shape[1:]))
    return torch.from_numpy(np.stack(selected))


def resize_road(road, height, width):
    """Sample the road union at the exact coordinates used by GCDID."""
    if not road.is_cuda:
        raise ValueError('Road sampling requires CUDA tensors')
    native_height, native_width = road.shape
    y = nearest_axis(native_height, height, road.device)
    x = nearest_axis(native_width, width, road.device)
    return road[y[:, None], x[None, :]]


def save_road(path, road):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, road=road.cpu().numpy())
    temporary.replace(path)


@contextmanager
def road_cache(cache, source, height, width, split, keys):
    """Publish the completion marker only after the requested frames are present."""
    config = dict(format=ROAD_FORMAT, source=str(Path(source).resolve()), height=height, width=width,
                  split=split, frames=len(keys), keys_sha256=sha256('\n'.join(keys).encode()).hexdigest())
    frames = cache / 'frames'
    frames.mkdir(parents=True, exist_ok=True)
    with (cache / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config_path = cache / 'config.json'
        text = json.dumps(config, indent=2) + '\n'
        if config_path.is_file():
            if json.loads(config_path.read_text()) != config:
                raise ValueError('Road cache configuration mismatch: ' + str(cache))
        else:
            temporary = config_path.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(text)
            temporary.replace(config_path)
        yield frames
        if not all((frames / (key + '.npz')).is_file() for key in keys):
            raise RuntimeError('Incomplete GCIDC road cache: ' + str(cache))
        summary = cache / 'summary.json'
        if not summary.is_file():
            temporary = summary.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(text)
            temporary.replace(summary)
        print('GCIDC roads complete:', cache, flush=True)
