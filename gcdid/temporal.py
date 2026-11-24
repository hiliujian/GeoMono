"""Cache locations and formats for training masks."""
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / 'cache' / 'geometry'
DETECTOR = 'GCDID ground parallax'
ROAD_FORMAT = 'road-union-nearest-exact'


def mask_cache(height, width, split=None):
    suffix = '_' + split if split and split != 'eigen_zhou' else ''
    return CACHE / f'GCDID{suffix}_{height}x{width}'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('height', type=int)
    parser.add_argument('width', type=int)
    parser.add_argument('split', nargs='?')
    args = parser.parse_args()
    print(mask_cache(args.height, args.width, args.split))
