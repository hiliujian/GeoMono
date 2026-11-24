"""Attach cached GCDID selections to training samples."""
from contextlib import ExitStack
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from .instances import frame_key
from .riders import cache_stamp
from .temporal import DETECTOR, FLOW_FORMAT, ROAD_FORMAT


_GCIDC_CACHE_FIELDS = ('region_cell_idx', 'region_cell_ptr', 'gcdid_selected')


class MotionDataset(Dataset):
    def __init__(self,base,cache,road_cache,flow_cache):
        self.base_dataset=base;self.cache=Path(cache)/'frames'
        self.flow_cache=Path(flow_cache)/'frames'
        config=Path(flow_cache)/'config.json'
        if not config.is_file():
            raise FileNotFoundError("Prepare GCIDC correspondence cache with run_train.sh: "+str(flow_cache))
        meta=json.loads(config.read_text())
        if (meta.get("format"),meta.get("height"),meta.get("width"))!=(FLOW_FORMAT,base.height,base.width):
            raise ValueError("Incompatible GCIDC flow cache: "+str(flow_cache))
        self.road_cache=Path(road_cache)/'frames' 
        seal=Path(road_cache)/'summary.json'
        if not seal.is_file():raise ValueError('Prepare the GCIDC road cache first: '+str(road_cache))
        road_info=json.loads(seal.read_text())
        if road_info.get('format')!=ROAD_FORMAT or (road_info.get('height'),road_info.get('width'))!=(base.height,base.width):
            raise ValueError('GCIDC road cache configuration mismatch: '+str(road_cache))
        summary=Path(cache)/'summary.json'
        if not summary.exists():
            raise ValueError('Prepare the GCDID cache first: '+str(cache))
        metadata=json.loads(summary.read_text())
        detector = metadata.get('detector')
        if detector != DETECTOR or meta.get('detector') != detector:
            raise ValueError(f'Incompatible GCDID mask and GCIDC flow detectors at {cache} and {flow_cache}')
        if (metadata.get('height'),metadata.get('width'))!=(base.height,base.width):
            raise ValueError('GCDID cache and training resolution differ: '+str(cache))
        self.rider_frames = set()
        requested = Path(cache) / 'requested_frames.json'
        flow_summary = Path(flow_cache) / 'summary.json'
        if requested.is_file() and flow_summary.is_file():
            source = sha256(summary.read_bytes() + requested.read_bytes()).hexdigest()
            self.rider_stamp = cache_stamp(source)
            cached = json.loads(flow_summary.read_text())
            if cached.get('rider_stamp') == self.rider_stamp:
                self.rider_frames = set(cached['rider_frames'])

    def __len__(self):return len(self.base_dataset)

    def __getitem__(self,index):
        base=self.base_dataset;original=base.get_color;flipped=[]
        def observe_flip(*args,**kwargs):
            flipped.append(bool(args[3] if len(args)>3 else kwargs.get('do_flip',False)))
            return original(*args,**kwargs)
        base.get_color=observe_flip
        try:inputs=base[index]
        finally:base.get_color=original
        key=frame_key(base.filenames[index]);path=self.cache/(key+'.npz')
        if not path.exists():raise FileNotFoundError('Prepare GCDID cache first: '+str(path))
        with ExitStack() as opened:
            original = opened.enter_context(np.load(path, allow_pickle=False))
            has_motion = bool(int(original["n_selected"]))
            record = (opened.enter_context(np.load(self.flow_cache/(key+'.npz'), allow_pickle=False))
                      if has_motion else None)
            fused = key in self.rider_frames
            if fused and (record is None or 'rider_stamp' not in record.files
                          or str(record['rider_stamp']) != self.rider_stamp):
                raise ValueError('GCIDC rider record does not match the prepared cache: '+key)
            z = record if fused else original
            prefix = 'rider_' if fused else ''
            flip = bool(flipped[0])
            suffix = '_flip' if flip else ''
            grid = z[prefix+'grid'+suffix]
            if grid.shape!=(base.height,base.width):raise ValueError('GCDID cache and training resolution differ')
            inputs['dynamic_mask']=torch.from_numpy(z[prefix+'moving'+suffix])
            inputs['gcdid_grid']=torch.from_numpy(grid)
            inputs['region_cell_idx']=torch.from_numpy(z[prefix+'cells'+suffix])
            inputs['region_cell_ptr']=torch.from_numpy(z[prefix+'ptr'+suffix])
            inputs['gcdid_selected']=torch.from_numpy(z[prefix+'selected'])
            inputs['gcdid_counts']=torch.tensor([len(original['groups']),int(z[prefix+'n_selected'])],dtype=torch.int32)
            inputs['gcidc_anchor_mask'] = (torch.from_numpy(record['rider_anchor'+suffix])
                                           if fused
                                           else torch.ones(1,dtype=torch.bool).expand(base.height,base.width))
            if record is not None:
                inputs["gcidc_flow"]=torch.from_numpy(record["flow"])
                inputs["gcidc_intrinsics"]=torch.from_numpy(record["intrinsics"])
            else:
                inputs["gcidc_flow"]=torch.zeros(1).expand(4,2,base.height,base.width)
                inputs["gcidc_intrinsics"]=torch.tensor([1.,1.,0.,0.],dtype=torch.float64)
        with np.load(self.road_cache/(key+'.npz')) as roads:
            road=roads['road'].copy()
        if road.shape!=grid.shape:raise ValueError('GCIDC road mask shape mismatch: '+key)
        inputs['gcidc_road']=torch.from_numpy(road)
        inputs['gcidc_flipped']=torch.tensor(flip,dtype=torch.bool)
        inputs['frame_key']=key
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
    cell_sizes, pointer_sizes, instance_sizes = sizes.unbind(1)
    # Preserve the existing first-frame eligibility for GCIDC supervision.
    if not pointers.numel():
        pointers = pointers.new_zeros(1)
    last = pointers[(pointer_sizes[0] - 1).clamp_min(0).view(1)].squeeze(0)
    inputs['gcidc_batch_enabled'] = ((pointer_sizes[0] > 1) & (cell_sizes[0] > 0)
                                      & (last == cell_sizes[0]))

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
