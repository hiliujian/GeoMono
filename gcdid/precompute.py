"""Build training masks from the GCDID ground-motion detector."""

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcdid.ground.settings import H, W
from gcdid.temporal import DETECTOR, mask_cache

DETECTION_SIZE = (H, W)
DEFAULT_ROADS = str(ROOT / 'data/road_masks')


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT / 'data/kitti'))
    parser.add_argument('--candidates', default=str(ROOT / 'candidates/kitti_dynamic_mask'))
    parser.add_argument('--road-masks', default=DEFAULT_ROADS)
    parser.add_argument('--split', default='eigen_zhou')
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--gcidc-road-cache', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--frames', type=Path)
    parser.add_argument('--stage', choices=['all', 'grids', 'evidence', 'pool'], default='all')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('--shard must be in [0, --shards)')
    if args.gcidc_road_cache and args.stage != 'all':
        parser.error('GCIDC cache preparation requires --stage all')
    return args


def atomic_json(path, data):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def cached_settings(value):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def same_settings(stored, cfg):
    return isinstance(stored, dict) and all(stored.get(k) == v for k, v in cfg.items())


def check_cache_detector(cache):
    """Check the cache identity before writing frame records."""
    for name in ('config.json', 'summary.json'):
        meta = cache / name
        if meta.is_file():
            settings = json.loads(meta.read_text())
            if settings.get('geometry_backend') != 'cuda':
                raise ValueError(f'{cache} holds a different geometry backend. Use a new CUDA cache directory.')
            previous = settings.get('detector')
            if previous != DETECTOR:
                raise ValueError(f'{cache} contains masks for detector {previous!r}. '
                                 f'Use a separate --cache directory for {DETECTOR}.')


def prepare_grid(file, height, width):
    from gcdid.instances import candidate_grids, region_cells
    grid, flip, groups = candidate_grids(file, height, width)
    cells, ptr = region_cells(grid, len(groups))
    fcells, fptr = region_cells(flip, len(groups))
    return dict(grid=grid, grid_flip=flip, groups=groups, cells=cells, ptr=ptr, cells_flip=fcells, ptr_flip=fptr)


def valid_evidence(record, threshold):
    fields = ('threshold', 'score', 'scores', 'road_h', 'road_variance','coverage', 'contact_columns', 'contact',
              'selected', 'moving', 'moving_flip', 'n_selected', 'selected_pixels', 'flow_available')
    return all(k in record for k in fields) and float(record['threshold']) == threshold


def requested_keys(args):
    from gcdid.instances import frame_key

    source = args.frames or ROOT / 'splits' / args.split / 'train_files.txt'
    lines = (line.strip() for line in source.read_text().splitlines())
    keys = list(dict.fromkeys(line if '__' in line else frame_key(line) for line in lines if line))
    return keys[:args.limit] if args.limit else keys


def reference_keys(split):
    from gcdid.instances import frame_key
    folder = ROOT / 'splits' / split

    def read(path):
        return list(dict.fromkeys(frame_key(line) for line in path.read_text().splitlines() if line.strip()))
    training = read(folder / 'train_files.txt')
    manifest = folder / 'motion_reference.txt'
    reference = read(manifest) if manifest.exists() else training
    if not reference or not set(reference).issubset(training):
        raise ValueError('Motion calibration images must belong to the training split')
    return reference


def calibration_signature(reference, args):
    files = [ROOT / 'gcdid' / name for name in (
        'flow.py', 'instances.py', 'sampling.py', 'ground/camera.py', 'ground/cuda_inputs.py',
        'ground/cuda_ops.py', 'ground/detector.py', 'ground/geometry_kernels.cu', 'ground/reference.py',
        'ground/road_dispersion.json', 'ground/settings.py', 'ground/visual_pose.py')]
    waft = ROOT / 'third_party/WAFT'
    files += sorted((waft / 'model').rglob('*.py'))
    files.extend([waft / 'config/a1/tar-c-t.json', ROOT / 'models/waft/tar-c-t.pth'])
    from gcdid.instances import image_path, at_frame
    assets = set()
    data = Path(args.data)
    for key in reference:
        date, _, frame, _ = key.split('__')
        optics = data / date / 'camera_intrinsics.json'
        assets.add(optics if optics.is_file() else data / date / 'calib_cam_to_cam.txt')
        assets.add(Path(args.candidates) / (key + '.npz'))
        assets.add(Path(args.road_masks) / (key + '.npz'))
        assets.update(image_path(data, at_frame(key, int(frame) + step)) for step in (-1, 0, 1))
    files.extend(sorted(assets))
    digest = sha256('\n'.join(reference).encode())
    for file in files:
        digest.update(file.stat().st_size.to_bytes(8, 'little'))
        with file.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def build(args, road_frames=None, keys=None):
    import torch
    from gcdid.ground.cuda_inputs import inputs, optical_matrix
    from gcdid.ground.detector import measure, empty_evidence
    from gcdid.ground.reference import fit_reference
    from gcdid.flow import FlowModel
    from gcdid.instances import at_frame, candidate_grids, image_path
    from gcdid.road_cache import read_masks, resize_road, save_road

    if not torch.cuda.is_available():
        raise RuntimeError('GCDID cache preparation requires CUDA')
    torch.set_num_threads(1)
    h, w = args.height, args.width
    cache = args.cache or mask_cache(h, w, args.split)
    frames = cache / 'frames'
    frames.mkdir(parents=True, exist_ok=True)
    keys = requested_keys(args) if keys is None else keys
    reference = reference_keys(args.split) if args.stage != 'grids' else []
    cfg = dict(
        height=h, width=w, data=str(Path(args.data).resolve()), candidates=str(Path(args.candidates).resolve()),
        road_masks=str(Path(args.road_masks).resolve()), detector=DETECTOR, detector_size=list(DETECTION_SIZE),
        geometry_backend='cuda', grid_sha256=sha256((ROOT / 'gcdid/instances.py').read_bytes() +
                                                  (ROOT / 'gcdid/sampling.py').read_bytes()).hexdigest(),
        calibration_source='unlabelled_rgb', pose_source='visual_road_background_pose',
        motion_evidence='stationary_column_reprojection',
        offsets=[-1, 0, 1], waft='a1/tar-c-t',
    )
    grid_settings = dict(cfg)
    if reference:
        cfg['calibration_sha256'] = calibration_signature(reference, args)
    ident = json.dumps(cfg, sort_keys=True)
    summary = cache / 'summary.json'
    manifest = cache / 'requested_frames.json'

    def prepare_road(key, detection_needed=False):
        if road_frames is None:
            return None
        destination = road_frames / (key + '.npz')
        write = not destination.is_file()
        if not (write or detection_needed):
            return None
        masks = read_masks(Path(cfg['road_masks']) / (key + '.npz')).cuda()
        native_road = masks.bool().any(0)
        detector_road = resize_road(native_road, *DETECTION_SIZE) if detection_needed else None
        if write:
            training_road = detector_road
            if detector_road is None or (h, w) != DETECTION_SIZE:
                training_road = resize_road(native_road, h, w)
            save_road(destination, training_road)
        if detection_needed:
            native_height, native_width = native_road.shape
            K = optical_matrix(cfg['data'], key, native_width, native_height, masks.device)
            return detector_road, K

    model = None

    def flow_model():
        nonlocal model
        if model is None:
            model = FlowModel(ROOT)
        return model

    check_cache_detector(cache)
    fitted = cache / 'motion_model.json'
    warmed = {}
    threshold = None
    if args.stage != 'grids':
        import fcntl
        with (cache / '.motion-calibration.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(fitted.read_text()) if fitted.is_file() else {}
            if state.get('settings') != cfg:
                print('Fitting the motion boundary from unlabelled training images', flush=True)
                result, warmed = fit_reference(reference, cfg, flow_model)
                state = dict(settings=cfg, **result)
                atomic_json(fitted, state)
            threshold = state['cut']
    if args.stage == 'pool' and summary.exists():
        summary.unlink()
    if args.stage in ('all', 'grids') and summary.exists():
        previous = json.loads(summary.read_text())
        complete = (bool(previous.get('sample_audit')) and same_settings(previous, cfg)
                    and manifest.exists() and json.loads(manifest.read_text()) == keys
                    and set(keys).issubset(p.stem for p in frames.glob('*.npz')))
        if complete and (args.stage == 'grids' or
                         previous.get('motion_model_sha256') == sha256(fitted.read_bytes()).hexdigest()):
            if road_frames is not None:
                for key in keys:
                    prepare_road(key)
            print('Using completed GCDID cache', cache, flush=True)
            return
        if not complete or args.stage == 'all':
            summary.unlink()
    clock = time.perf_counter()

    def save(path, record):
        tmp = path.with_suffix(f'.{os.getpid()}.npz')
        arrays = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in record.items()}
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)

    def replay(key, grid, groups, road_input=None):
        number = int(key.split('__')[2])
        paths = [image_path(cfg['data'], at_frame(key, number + k)) for k in (-1, 0, 1)]
        missing = next((p for p in paths if not p.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f'GCDID triplet for {key} is missing {missing}')
        groups = torch.as_tensor(groups, device='cuda')
        if not len(groups):
            return empty_evidence(len(groups), groups.device)
        with torch.inference_mode():
            flow, endpoint_flow = flow_model().triplet(paths, *DETECTION_SIZE, endpoints=True)
            if road_input is None:
                road_input = inputs(cfg['road_masks'], key, flow.device, cfg['data'])
            road, intrinsics = road_input
            out = measure(road, intrinsics, flow, torch.as_tensor(grid, device=flow.device), groups, endpoint_flow)
            out['selected'] = out['score'] > threshold
            return out

    if args.stage in ('all', 'grids'):
        atomic_json(cache / 'config.json', cfg)
        atomic_json(manifest, keys)
    work = keys[args.shard::args.shards] if args.stage == 'evidence' else keys
    for i, key in enumerate(work if args.stage != 'pool' else []):
        path = frames / (key + '.npz')
        candidate_path = Path(cfg['candidates']) / (key + '.npz')
        record = None
        if path.exists():
            with np.load(path) as z:
                stored = cached_settings(str(z['geometry_key'])) if 'geometry_key' in z else None
                if same_settings(stored, grid_settings):
                    if args.stage == 'grids' or (same_settings(stored, cfg) and valid_evidence(z, threshold)):
                        prepare_road(key)
                        continue
                    record = {name: z[name] for name in ('grid', 'grid_flip', 'groups', 'cells', 'ptr',
                                                        'cells_flip', 'ptr_flip')}
                    record['geometry_key'] = ident
        if record is None:
            record = prepare_grid(candidate_path, h, w)
            record['geometry_key'] = ident
        if args.stage == 'grids':
            save(path, record)
            continue

        grid = torch.as_tensor(record['grid'], device='cuda').long()
        flip = torch.as_tensor(record['grid_flip'], device=grid.device).long()
        detector_grid = grid
        groups = torch.as_tensor(record['groups'], device=grid.device)
        if (h, w) != DETECTION_SIZE and len(groups):
            detector_grid = candidate_grids(candidate_path, *DETECTION_SIZE)[0]
        evidence = warmed.pop(key, None)
        road_input = prepare_road(key, detection_needed=evidence is None and bool(len(groups)))
        if evidence is None:
            evidence = replay(key, detector_grid, groups, road_input=road_input)
        else:
            evidence['selected'] = evidence['score'] > threshold
        selected = evidence['selected']
        flags = torch.cat((selected.new_zeros(1), selected))
        moving = flags[grid + 1]
        record.update(evidence, threshold=threshold, flow_available=True, moving=moving,
                      moving_flip=flags[flip + 1], n_selected=evidence['selected'].sum(),
                      selected_pixels=moving.sum())
        save(path, record)
        if i % 100 == 0 or i == len(work) - 1:
            print(f'GCDID {i+1}/{len(work)} {time.perf_counter()-clock:.1f}s', flush=True)

    if args.stage in ('all', 'pool'):
        from gcdid.cache_audit import audit_samples
        count = px = unavailable = 0
        positive_keys = []
        for key in keys:
            with np.load(frames / (key + '.npz')) as z:
                if not same_settings(cached_settings(str(z['geometry_key'])), cfg) or not valid_evidence(z, threshold):
                    raise ValueError('Frame settings differ from requested GCDID settings: ' + key)
                n_selected = int(z['n_selected'])
                count += n_selected
                if n_selected:
                    positive_keys.append(key)
                px += int(z['selected_pixels'])
                unavailable += int(not bool(z['flow_available']))
        if unavailable:
            raise ValueError(f'GCDID cache has {unavailable} incomplete image triplets')
        checked = audit_samples(frames, keys, positive_keys, cfg, replay)
        result = dict(cfg, sample_audit=checked, training_frames=len(keys), selected_instances=count,
                      selected_pixels=px, missing_triplets=unavailable, motion_model_sha256=sha256(fitted.read_bytes()).hexdigest(),
                      seconds=time.perf_counter()-clock)
        atomic_json(cache / 'summary.json', result)
        print('GCDID cache ready', cache, result, flush=True)


def main():
    args = arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    from gcdid.road_cache import road_cache

    keys = requested_keys(args)
    roads = (road_cache(args.gcidc_road_cache, args.road_masks, args.height, args.width, args.split, keys)
             if args.gcidc_road_cache is not None else nullcontext())
    with roads as road_frames:
        build(args, road_frames=road_frames, keys=keys)


if __name__ == '__main__':
    main()
