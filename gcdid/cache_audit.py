"""Audit the GCDID geometry decisions, sampling and cached mask orientations."""

from pathlib import Path
from random import SystemRandom

import numpy as np
import torch
from .sampling import nearest_axis


def _native_owner(path):
    """Reconstruct the original silhouettes in candidate order for the offline check."""
    with np.load(path) as src:
        height, width = int(src["height"]), int(src["width"])
        owner = torch.full((height, width), -1, dtype=torch.int32, device="cuda")
        packed = torch.as_tensor(src["masks_bits"], device=owner.device)
        shifts = torch.arange(7, -1, -1, device=owner.device)
        offsets = src["masks_off"]
        for i, (x0, y0, x1, y1) in enumerate(src["boxes"].tolist()):
            crop = owner[int(y0):int(y1)+1, int(x0):int(x1)+1]
            block = packed[int(offsets[i]):int(offsets[i+1])]
            bits = ((block[:, None] >> shifts) & 1).flatten()[:crop.numel()]
            crop[(crop < 0) & bits.reshape_as(crop).bool()] = i
        return owner, len(src["labels"])


def resize_owner(original, height, width):
    y = nearest_axis(original.shape[0], height, original.device)
    x = nearest_axis(original.shape[1], width, original.device)
    return original[y[:, None], x]


def audit_samples(frames, keys, positive_keys, settings, replay):
    """Replay ground evidence on independently rasterized candidates and both flips."""
    h, w = settings['height'], settings['width']
    detect_h, detect_w = settings['detector_size']
    rng = SystemRandom()
    picked = []
    for sides in (('l', '2'), ('r', '3')):
        choices = [k for k in keys if k.rsplit('__', 1)[-1] in sides]
        if choices:
            picked.append(rng.choice(choices))
    remaining = sorted(set(positive_keys or keys).difference(picked))
    picked.extend(rng.sample(remaining, min(1, len(remaining))))
    rows = []
    for key in picked:
        native, count = _native_owner(Path(settings['candidates']) / (key + '.npz'))
        detect_grid = resize_owner(native, detect_h, detect_w)
        with np.load(frames / (key + '.npz')) as frame:
            def cached(field):
                return torch.as_tensor(frame[field], device=native.device)
            groups = frame['groups']
            if len(groups) != count:
                raise ValueError(f'GCDID instance count differs on {key}')
            evidence = replay(key, detect_grid, groups)
            for name, expected in evidence.items():
                value = cached(name)
                if value.shape != expected.shape or not torch.allclose(value, expected, rtol=1e-4, atol=1e-4, equal_nan=True):
                    raise ValueError(f'GCDID {name} differs on {key}')
            selected = cached('selected')
            decision = cached('score') > float(frame['threshold'])
            if not torch.equal(selected, decision):
                raise ValueError(f'GCDID final decision differs on {key}')
            flags = torch.cat((selected.new_zeros(1), selected))
            for suffix, original in (('', native), ('_flip', native.flip((1,)))):
                grid = resize_owner(original, h, w)
                if not torch.equal(cached('grid' + suffix), grid):
                    raise ValueError(f'GCDID instance positions differ on {key}{suffix}')
                if not torch.equal(cached('moving' + suffix), flags[grid + 1]):
                    raise ValueError(f'GCDID motion mask differs on {key}{suffix}')
            rows.append(dict(frame=key, reference='ground_triplet_replay',
                             candidates=count, selected=int(selected.sum())))
    return rows
