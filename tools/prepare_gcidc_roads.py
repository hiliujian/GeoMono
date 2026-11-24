"""Prepare GCIDC road masks when a GCDID cache already exists."""
import argparse
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcdid.instances import frame_key
from gcdid.road_cache import read_masks, resize_road, road_cache, save_road


class Sources(Dataset):
    def __init__(self, keys, source):
        self.keys = keys
        self.source = source

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        return key, read_masks(self.source / (key + '.npz'))


def identity(value):
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/road_masks')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--split', default='eigen_zhou')
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GCIDC road preparation requires CUDA')
    lines = (args.project / 'splits' / args.split / 'train_files.txt').read_text().splitlines()
    keys = [frame_key(line) for line in lines if line.strip()]
    cache = args.cache or args.project / 'cache/geometry' / f'GCIDC_roads_{args.height}x{args.width}'
    with road_cache(cache, args.source, args.height, args.width, args.split, keys) as frames:
        pending = [key for key in keys if not (frames / (key + '.npz')).is_file()]
        loader = DataLoader(Sources(pending, args.source), batch_size=None, collate_fn=identity,
                            num_workers=args.workers, pin_memory=True)
        start = time.monotonic()
        for index, (key, masks) in enumerate(loader, 1):
            road = masks.cuda(non_blocking=True).bool().any(0)
            result = resize_road(road, args.height, args.width)
            save_road(frames / (key + '.npz'), result)
            if index % 500 == 0 or index == len(pending):
                print(f'GCIDC roads {index}/{len(pending)} {time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
