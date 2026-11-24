"""Read and write the GCIDC correspondence cache shared with GCDID preparation."""
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .flow import FlowModel
from .ground.settings import H, W
from .instances import at_frame, image_path
from .temporal import DETECTOR, FLOW_FORMAT

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


@contextmanager
def correspondence_cache(cache, data, height, width):
    cache.mkdir(parents=True, exist_ok=True)
    config = cache / "config.json"
    weights = ROOT / "models/waft/tar-c-t.pth"
    checksum = sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(chunk)
    spec = dict(format=FLOW_FORMAT, data=str(Path(data).resolve()), height=height, width=width,
                weights_sha256=checksum.hexdigest(), detector=DETECTOR)
    with (cache / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if config.exists():
            if json.loads(config.read_text()) != spec:
                raise ValueError("GCIDC flow cache configuration mismatch: " + str(cache))
        else:
            write_json(config, spec)
        frames = cache / "frames"
        frames.mkdir(exist_ok=True)
        yield frames, spec


def save_flow(path, flow, data, key, center, height, width):
    if (height, width) != (H, W):
        flow = F.interpolate(flow, (height, width), mode="bilinear", align_corners=False)
        flow = flow * flow.new_tensor([width / W, height / H])[None, :, None, None]
    from .ground.camera import correspondence_intrinsics
    with Image.open(center) as image:
        native_width,native_height=image.size
    intrinsics=correspondence_intrinsics(data,key,width,height,native_width,native_height,flow.device)
    if not torch.isfinite(flow).all():
        raise ValueError("Nonfinite WAFT flow in " + key)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, flow=flow.cpu().numpy(), intrinsics=intrinsics.cpu().numpy())
    temp.replace(path)


@torch.inference_mode()
def complete_flow(masks, folder, spec, selected_keys=None):
    manifest = (masks / "requested_frames.json").read_bytes()
    summary = (masks / "summary.json").read_bytes()
    source_hash = sha256(summary + manifest).hexdigest()
    seal = folder.parent / "summary.json"
    if seal.exists():
        previous = json.loads(seal.read_text())
        selected = previous.get("keys", [])
        if (previous.get("source_hash") == source_hash
                and len(selected) == previous.get("selected_frames")
                and all((folder / (key + ".npz")).is_file() for key in selected)):
            print("Using completed GCIDC flow cache", folder.parent, flush=True)
            return
    keys = json.loads(manifest)
    if selected_keys is None:
        selected_keys = []
        for key in keys:
            with np.load(masks / 'frames' / (key + '.npz')) as record:
                if int(record['n_selected']):
                    selected_keys.append(key)
    model = None
    data = Path(spec["data"])
    for key in selected_keys:
        dest = folder / (key + ".npz")
        if dest.is_file():
            continue
        if model is None:
            print("Computing missing GCIDC correspondences", flush=True)
            model = FlowModel(ROOT)
        number = int(key.split("__")[2])
        paths = [image_path(data, at_frame(key, number + offset)) for offset in (-1, 0, 1)]
        flow = model.triplet(paths, H, W)
        save_flow(dest, flow, data, key, paths[1], spec["height"], spec["width"])
    write_json(seal, dict(**spec, frames=len(keys), selected_frames=len(selected_keys),
                          keys=selected_keys, source_hash=source_hash))
    print("GCIDC flow ready", len(selected_keys), folder.parent, flush=True)
