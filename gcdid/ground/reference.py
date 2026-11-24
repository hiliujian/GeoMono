"""Estimate the motion decision from optical measurements on unlabelled training images."""
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from ..instances import GROUPS, at_frame, candidate_grids, image_path
from ..sampling import nearest_axis
from .cuda_inputs import optical_matrix
from .detector import measure
from .settings import H, W, ROAD


def _mask_blocks(path, height, width):
    """Stream mask planes from the existing semantic archive, preserving proposal order."""
    with ZipFile(path) as archive, archive.open('masks.npy') as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version in readers:
            shape, fortran, dtype = readers[version](stream)
            if not fortran and shape[1:] == (height, width) and dtype.kind in ('b', 'u'):
                for start in range(0, shape[0], 8):
                    n = min(8, shape[0] - start)
                    raw = stream.read(n * height * width * dtype.itemsize)
                    yield start, np.frombuffer(raw, dtype=dtype).copy().reshape(n, height, width)
                return
    with np.load(path, allow_pickle=False) as source:
        masks = source['masks']
        for start in range(0, len(masks), 8):
            yield start, np.ascontiguousarray(masks[start:start + 8])


def reference_grid(path, candidates):
    """Give each calibration pixel to its smallest covering native proposal."""
    with np.load(path, allow_pickle=True) as source:
        labels = [str(s).lower() for s in source['labels']]
        height, width = int(source['image_height']), int(source['image_width'])
    device = candidates.device
    y = nearest_axis(height, H, device)
    x = nearest_axis(width, W, device)
    count = len(labels)
    background = (height * width + 1) * (count + 1)
    minimum = torch.full_like(candidates, background, dtype=torch.long)
    road = torch.zeros_like(candidates, dtype=torch.bool)
    road_classes = torch.tensor([label in ROAD for label in labels], device=device, dtype=torch.bool)
    for start, block in _mask_blocks(path, height, width):
        masks = torch.as_tensor(block, device=device).bool()
        ids = torch.arange(start, start + len(masks), device=device)
        sampled = masks[:, y[:, None], x]
        road |= (sampled & road_classes[ids, None, None]).any(0)
        # The row index breaks equal-area ties in the native proposal order.
        rank = masks.sum((1, 2)) * (count + 1) + ids
        minimum = torch.minimum(minimum, torch.where(sampled, rank[:, None, None], background).amin(0))
    if not any(label in GROUPS for label in labels):
        return candidates, road, width, height
    owner = minimum.remainder(count + 1)
    eligible = torch.tensor([label in GROUPS for label in labels] + [False], device=device)
    visible = (candidates >= 0) & (minimum < background) & eligible[owner]
    ids = owner[visible].unique()
    remap = torch.full((count + 1,), -1, device=device, dtype=torch.int16)
    remap[ids] = torch.arange(len(ids), device=device, dtype=torch.int16)
    return torch.where(visible, remap[owner], -1), road, width, height


def same_partition(first, second):
    supported = first >= 0
    if not torch.equal(supported, second >= 0):
        return False
    a, b = first[supported].long(), second[supported].long()
    correspondence = torch.stack((a, b), dim=1).unique(dim=0)
    return len(correspondence) == a.unique().numel() == b.unique().numel()


def fit_partition(values):
    scores = values.sort().values
    n = len(scores)
    if n < 2 or not scores.isfinite().all() or scores[0] == scores[-1]:
        raise ValueError('The training reference needs two distinguishable motion scores')
    count = torch.arange(1, n, device=scores.device, dtype=scores.dtype)
    left = scores.cumsum(0)[:-1]
    low, high = left / count, (scores.sum() - left) / (n - count)
    objective = count * (n - count) * (high - low).square()
    index = objective.masked_fill(scores[:-1] == scores[1:], -1).argmax()
    return dict(centroids=[float(low[index]), float(high[index])],
                cut=float((low[index] + high[index]) / 2), candidates=n)


@torch.inference_mode()
def fit_reference(keys, settings, model):
    samples, measurements = [], {}
    for key in keys:
        grid, _, groups = candidate_grids(Path(settings['candidates']) / (key + '.npz'), H, W)
        if not len(groups) or not (grid >= 0).any():
            continue
        native, road, width, height = reference_grid(Path(settings['road_masks']) / (key + '.npz'), grid)
        intrinsics = optical_matrix(settings['data'], key, width, height, grid.device)
        frame = int(key.split('__')[2])
        paths = [image_path(settings['data'], at_frame(key, frame + offset)) for offset in (-1, 0, 1)]
        flow, span = model().triplet(paths, H, W, endpoints=True)
        evidence = measure(road, intrinsics, flow, grid, torch.as_tensor(groups, device=grid.device), span)
        measurements[key] = evidence
        if same_partition(grid, native):
            samples.append(evidence['score'][grid[grid >= 0].long().unique()])
        elif (native >= 0).any():
            size = int(native.max()) + 1
            observation = measure(road, intrinsics, flow, native, native.new_zeros(size), span)
            samples.append(observation['score'][native[native >= 0].long().unique()])
    if not samples:
        raise ValueError('No visible semantic instances in the unlabelled training reference')
    return dict(fit_partition(torch.cat(samples)), frames=len(keys)), measurements
