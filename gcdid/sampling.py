"""CUDA sampling at the RGB and instance coordinates used to calibrate GCDID."""
from functools import lru_cache
from math import ceil, gcd

import torch


@lru_cache(maxsize=64)
def nearest_axis(native, size, device):
    divisor = gcd(native, size)
    denominator = size // divisor
    if native // divisor % 2 or denominator & (denominator - 1) == 0:
        return (2 * torch.arange(size, device=device) + 1) * native // (2 * size)
    # At exact pixel boundaries, reproduce Pillow's successive floating-point
    # additions. Build this one-dimensional lookup once per camera resolution.
    positions = torch.empty(size, dtype=torch.float64, device=device)
    step = native / size
    positions[0] = step * 0.5
    for index in range(1, size):
        torch.add(positions[index - 1], step, out=positions[index])
    return positions.long()


@lru_cache(maxsize=64)
def _lanczos_axis(native, size, device):
    ratio = native / size
    span = max(1.0, ratio)
    center = (torch.arange(size, device=device, dtype=torch.float64) + 0.5) * ratio - 0.5
    radius = ceil(3 * span)
    index = center.floor().long()[:, None] + torch.arange(-radius, radius + 1, device=device)
    delta = (index - center[:, None]) / span
    weights = torch.sinc(delta) * torch.sinc(delta / 3)
    weights *= (index >= 0) & (index < native) & (delta.abs() < 3)
    weights /= weights.sum(-1, keepdim=True)
    # The 8-bit RGB filter uses signed coefficients with 22 fractional bits.
    weights = weights.sign() * (weights.abs() * (1 << 22) + 0.5).floor()
    return index.clamp(0, native - 1), weights.int()


def _resize_axis(values, size):
    if values.shape[-1] == size:
        return values
    index, weights = _lanczos_axis(values.shape[-1], size, values.device)
    sums = (values[..., index] * weights).sum(-1, dtype=torch.int32)
    return ((sums + (1 << 21)) >> 22).clamp_(0, 255)


def resize_rgb(rgb, height, width):
    """Filter native HWC bytes and return WAFT's contiguous CHW float input."""
    horizontal = _resize_axis(rgb.permute(2, 0, 1), width)
    output = _resize_axis(horizontal.transpose(-1, -2), height)
    return output.transpose(-1, -2).contiguous().float()
