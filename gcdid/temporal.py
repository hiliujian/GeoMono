"""Cache locations and formats for training masks."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_TMP = PROJECT_ROOT / 'tmp'
CACHE = PROJECT_TMP / 'cache' / 'geometry'
DETECTOR = 'GCDID ground parallax'
ROAD_FORMAT = 'road-union-nearest-exact'


def output_path(path):
    """Keep generated experiment files physically inside the project tmp directory."""
    directory = (PROJECT_ROOT / Path(path).expanduser()).resolve()
    if directory != PROJECT_TMP and PROJECT_TMP not in directory.parents:
        raise ValueError(f'Output directory must be inside {PROJECT_TMP}')
    return directory


def road_mask_cache(height, width):
    return CACHE / f'GCIDC_roads_{height}x{width}'


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
