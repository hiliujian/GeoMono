"""Join visible rider and cycle silhouettes using cached correspondences."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import os
from pathlib import Path
from shutil import copyfile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from gcidc import observed_cycle

from .instances import GROUPS, candidate_names, region_cells

HUMAN = ("person", "pedestrian", "rider", "cyclist", "bicyclist", "motorcyclist")
CYCLE = ("bicycle", "motorcycle", "motorbike")


def pair_roles(names, device):
    human = torch.tensor([name in HUMAN for name in names], device=device, dtype=torch.bool)
    cycle = torch.tensor([name in CYCLE for name in names], device=device, dtype=torch.bool)
    return human, cycle


def contact_edges(grid, selected, human, cycle):
    """Find where a human silhouette is supported from below by a cycle."""
    if not grid.is_cuda:
        raise ValueError("Rider pairing requires CUDA masks")
    n = len(selected)
    if not n or not (human.any() and cycle.any() and selected.any()):
        return None
    height, width = grid.shape
    y, x = torch.where(grid >= 0)
    idx = grid[y, x].long()
    area = torch.bincount(idx, minlength=n).clamp_min(1)
    xmin = x.new_full((n,), width).scatter_reduce_(0, idx, x, reduce="amin")
    xmax = x.new_full((n,), -1).scatter_reduce_(0, idx, x, reduce="amax")
    ymin = y.new_full((n,), height).scatter_reduce_(0, idx, y, reduce="amin")
    centre_x = torch.zeros(n, device=grid.device).scatter_add_(0, idx, x.float()) / area
    centre_y = torch.zeros_like(centre_x).scatter_add_(0, idx, y.float()) / area

    # Adjacent cell centres may touch across an edge or at a shared corner.
    delta = torch.arange(-1, 2, device=grid.device)
    px = torch.arange(width, device=grid.device)
    adjacent_x = px + delta[:, None]
    qx = adjacent_x.clamp(0, width - 1)
    first = grid[:-1][None].expand(len(delta), -1, -1).long()
    second = grid[1:][:, qx].transpose(0, 1).long()
    h = first.clamp_min(0)
    b = second.clamp_min(0)
    possible = ((first >= 0) & (second >= 0) & human[h] & cycle[b]
                & (selected[h] | selected[b])
                & (ymin[h] < ymin[b]) & (centre_y[h] < centre_y[b])
                & (centre_x[h] >= xmin[b]) & (centre_x[h] <= xmax[b])
                & (adjacent_x == qx)[:, None])
    if not possible.any():
        return None
    k, row, col = torch.where(possible)
    h, b = first[k, row, col], second[k, row, col]
    bx, by = qx[k, col], row + 1
    return h, b, row, col, by, bx


def confirm_edges(grid, n, flow, edges):
    """Return uniquely matched IDs with stable contact in both neighbours."""
    if flow.device != grid.device:
        raise ValueError("Rider pairing requires CUDA flow")
    h, b, row, col, by, bx = edges
    observed, extent = observed_cycle(flow[None], grid.new_zeros((1,), dtype=torch.bool))
    # Preserve proximity under both forward flows up to the measured return errors
    # and at most one pixel of combined raster quantization per coordinate.
    first_step = flow[::2][:, :, row, col]
    next_step = flow[::2][:, :, by, bx]
    displacement = torch.stack((bx - col, by - row))
    bound = displacement.abs() + 1 + extent[0, :, row, col] + extent[0, :, by, bx]
    agrees = ((displacement + next_step - first_step).abs() <= bound).all(dim=0).all(dim=0)
    agrees &= observed[0, row, col] & observed[0, by, bx]
    key = h * n + b
    seen = torch.bincount(key, minlength=n * n).reshape(n, n)
    stable = torch.bincount(key[agrees], minlength=n * n).reshape(n, n)
    # Most of the shared boundary must support the same pairing in both neighbours.
    evidence = torch.where(stable > seen - stable, stable, 0)
    best_human = (evidence == evidence.amax(0, keepdim=True)) & (evidence > 0)
    best_cycle = (evidence == evidence.amax(1, keepdim=True)) & (evidence > 0)
    unique = (best_human & best_cycle & (best_human.sum(0, keepdim=True) == 1)
              & (best_cycle.sum(1, keepdim=True) == 1))
    return unique.nonzero()


def mounted_pairs(grid, count, pairs):
    """Check that the wheel base extends below the rider and its centre lies under them."""
    height, width = grid.shape
    y, x = torch.where(grid >= 0)
    ids = grid[y, x].long()
    left = x.new_full((count,), width).scatter_reduce_(0, ids, x, reduce="amin")
    right = x.new_full((count,), -1).scatter_reduce_(0, ids, x, reduce="amax")
    bottom = y.new_full((count,), -1).scatter_reduce_(0, ids, y, reduce="amax")
    people, cycles = pairs.T
    cycle_centre = left[cycles] + right[cycles]
    contained = (2 * left[people] <= cycle_centre) & (cycle_centre <= 2 * right[people])
    visible = (bottom[cycles] > bottom[people]) & (bottom[cycles] < height - 1)
    return pairs[contained & visible]


def merge_pairs(grid, selected, pairs):
    """Preserve pixel ownership outside pairs and use only cycle pixels as anchors."""
    ids = torch.arange(len(selected), device=grid.device, dtype=torch.long)
    people, cycles = pairs.T
    chosen = selected.clone()
    chosen[cycles] = selected[people] | selected[cycles]
    chosen[people] = False
    ids[people] = cycles
    positive = grid >= 0
    merged = grid.clone()
    merged[positive] = ids[grid[positive].long()].to(grid.dtype)
    may_anchor = torch.ones_like(grid, dtype=torch.bool)
    forbidden = torch.zeros_like(chosen)
    forbidden[people] = True
    may_anchor[positive] = ~forbidden[grid[positive].long()]
    return merged, chosen, may_anchor


def cache_stamp(source_hash):
    return sha256(source_hash.encode() + Path(__file__).read_bytes()).hexdigest()


def write_rider_fields(path, fields, stamp):
    """Append fused inputs to the original correspondence record atomically."""
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with ZipFile(path) as archive:
            has_rider_fields = any(name.startswith("rider_") for name in archive.namelist())
        if has_rider_fields:
            with np.load(path, allow_pickle=False) as source, temporary.open("wb") as output:
                original = {key: source[key] for key in source.files if not key.startswith("rider_")}
                np.savez_compressed(output, **original)
        else:
            copyfile(path, temporary)
        with ZipFile(temporary, "a", compression=ZIP_DEFLATED) as archive:
            for name, value in dict(fields, stamp=np.asarray(stamp)).items():
                with archive.open(f"rider_{name}.npy", "w") as output:
                    np.lib.format.write_array(output, np.asarray(value), allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def prepare_riders(mask_cache, flow_frames, keys, stamp, candidates):
    """Store fused training instances in the existing optical-flow records."""
    if not torch.cuda.is_available():
        raise RuntimeError("Rider-cache preparation requires CUDA")

    def read(key):
        name = key + ".npz"
        with np.load(candidates / name, allow_pickle=False) as source:
            names = candidate_names(source["labels"])
        if not any(label in HUMAN for label in names) or not any(label in CYCLE for label in names):
            return key, None
        with np.load(mask_cache / "frames" / name, allow_pickle=False) as source:
            data = {field:source[field] for field in ("groups", "grid", "grid_flip", "selected")}
        if not np.array_equal(data["groups"], [GROUPS[name] for name in names]):
            raise ValueError("Rider labels and GCDID instance order differ on " + key)
        return key, (names, data)

    device = torch.device("cuda", torch.cuda.current_device())
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for offset in range(0, len(keys), 128):
            for key, sample in pool.map(read, keys[offset:offset+128]):
                if sample is None:
                    continue
                names, data = sample
                grid = torch.as_tensor(data["grid"], device=device)
                selected = torch.as_tensor(data["selected"], device=device).bool()
                human, cycle = pair_roles(names, device)
                edges = contact_edges(grid, selected, human, cycle)
                if edges is None:
                    continue
                path = flow_frames / (key + ".npz")
                with np.load(path, allow_pickle=False) as source:
                    if "rider_stamp" in source.files and str(source["rider_stamp"]) == stamp:
                        records.append(key)
                        continue
                    flow = torch.as_tensor(source["flow"], device=device)
                pairs = mounted_pairs(grid, len(selected), confirm_edges(grid, len(selected), flow, edges))
                if not pairs.numel():
                    continue
                fields = {"pairs": pairs.cpu().numpy()}
                for suffix in ("", "_flip"):
                    original = torch.as_tensor(data["grid"+suffix], device=device)
                    merged, active, anchor = merge_pairs(original, selected, pairs)
                    moving = torch.cat((selected.new_zeros(1), active))[merged.long()+1]
                    cells, ptr = region_cells(merged, len(active))
                    for field, value in (("grid", merged), ("moving", moving), ("anchor", anchor),
                                         ("cells", cells), ("ptr", ptr)):
                        fields[field+suffix] = value.cpu().numpy()
                fields["selected"] = active.cpu().numpy()
                fields["n_selected"] = int(active.sum())
                write_rider_fields(path, fields, stamp)
                records.append(key)
    return records
