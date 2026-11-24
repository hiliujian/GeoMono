"""Attach cached GCDID selections to training samples."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from .instances import frame_key
from .temporal import DETECTOR, ROAD_FORMAT


_GCIDC_CACHE_FIELDS = ('region_cell_idx', 'region_cell_ptr', 'gcdid_selected')


class MotionDataset(Dataset):
    def __init__(self, base, cache, road_cache):
        self.base_dataset = base
        self.cache = Path(cache) / 'frames'
        self.road_cache = Path(road_cache) / 'frames'
        shape = (base.height, base.width)
        for directory, kind, expected in ((Path(cache), 'detector', DETECTOR),
                                          (Path(road_cache), 'format', ROAD_FORMAT)):
            seal = directory / 'summary.json'
            if not seal.is_file():
                raise FileNotFoundError(f'Prepare the geometry cache with run_train.sh: {directory}')
            metadata = json.loads(seal.read_text())
            if metadata.get(kind) != expected or (metadata.get('height'), metadata.get('width')) != shape:
                raise ValueError(f'Incompatible geometry cache: {directory}')

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        base = self.base_dataset
        original = base.get_color
        flipped = []

        def observe_flip(*args, **kwargs):
            flipped.append(bool(args[3] if len(args) > 3 else kwargs.get('do_flip', False)))
            return original(*args, **kwargs)

        base.get_color = observe_flip
        try:
            inputs = base[index]
        finally:
            base.get_color = original
        key = frame_key(base.filenames[index])
        name = key + '.npz'
        flip = flipped[0]
        suffix = '_flip' if flip else ''
        with np.load(self.cache / name, allow_pickle=False) as z:
            grid = z['grid' + suffix]
            if grid.shape != (base.height, base.width):
                raise ValueError(f'GCDID mask shape mismatch: {key}')
            inputs['dynamic_mask'] = torch.from_numpy(z['moving' + suffix])
            inputs['gcdid_grid'] = torch.from_numpy(grid)
            inputs['region_cell_idx'] = torch.from_numpy(z['cells' + suffix])
            inputs['region_cell_ptr'] = torch.from_numpy(z['ptr' + suffix])
            inputs['gcdid_selected'] = torch.from_numpy(z['selected'])
            inputs['gcdid_counts'] = torch.tensor([len(z['groups']), int(z['n_selected'])], dtype=torch.int32)
        with np.load(self.road_cache / name, allow_pickle=False) as roads:
            road = roads['road']
        if road.shape != grid.shape:
            raise ValueError(f'GCIDC road mask shape mismatch: {key}')
        inputs['gcidc_road'] = torch.from_numpy(road)
        inputs['gcidc_flipped'] = torch.tensor(flip, dtype=torch.bool)
        inputs['frame_key'] = key
        return inputs


def motion_collate(batch):
    """Pack the raw cache records so workers only handle input and batching."""
    variable = {'frame_key', *_GCIDC_CACHE_FIELDS}
    out = default_collate([{key: value for key, value in sample.items() if key not in variable}
                           for sample in batch])
    out['frame_key'] = [sample['frame_key'] for sample in batch]
    out['gcidc_cache_sizes'] = torch.tensor(
        [[sample[name].numel() for name in _GCIDC_CACHE_FIELDS] for sample in batch],
        dtype=torch.long)
    for name in _GCIDC_CACHE_FIELDS:
        out[name] = torch.cat([sample[name] for sample in batch])
    return out


def prepare_gcidc_cells(inputs):
    """Gather and pad selected instances in cache order on the training GPU."""
    sizes = inputs.pop('gcidc_cache_sizes')
    indices, pointers, selected = (inputs.pop(name) for name in _GCIDC_CACHE_FIELDS)
    if not sizes.is_cuda or any(t.device != sizes.device for t in (indices, pointers, selected)):
        raise ValueError('GCIDC input preparation requires tensors on one CUDA device')

    cell_starts, pointer_starts, instance_starts = (sizes.cumsum(0) - sizes).unbind(1)
    instance_sizes = sizes[:, 2]

    device = sizes.device
    frames = torch.arange(len(sizes), device=device).repeat_interleave(
        instance_sizes, output_size=selected.numel())
    rows = torch.arange(selected.numel(), device=device) - instance_starts[frames]
    positions = pointer_starts[frames] + rows
    begin = pointers[positions]
    length = pointers[positions + 1] - begin
    keep = selected.bool() & (length > 0)
    frames, begin, length = frames[keep], begin[keep], length[keep]
    if length.numel():
        columns = torch.arange(int(length.max()), device=device)
        valid = columns[None] < length[:, None]
        addresses = (cell_starts[frames] + begin)[:, None] + columns
        cells = indices[torch.where(valid, addresses, 0)].long().masked_fill_(~valid, 0)
    else:
        # Keep a single invalid slot so an empty batch follows the same loss path.
        cells = torch.zeros((1, 1), device=device, dtype=torch.long)
        valid = torch.zeros_like(cells, dtype=torch.bool)
        frames = torch.zeros(1, device=device, dtype=torch.long)

    inputs['gcidc_cells'] = cells
    inputs['gcidc_cell_valid'] = valid
    inputs['gcidc_batch_index'] = frames
