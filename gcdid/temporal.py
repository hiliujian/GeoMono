"""Cache locations and data formats for GCDID masks and GCIDC correspondences."""
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / 'cache' / 'geometry'
DETECTOR = 'GCDID ground parallax'
FLOW_FORMAT = 'waft-four-directions'
ROAD_FORMAT = 'road-union-nearest-exact'


def mask_cache(height, width, split=None):
    suffix = '_' + split if split and split != 'eigen_zhou' else ''
    return CACHE / f'GCDID{suffix}_{height}x{width}'


def flow_cache(height, width, masks=None):
    masks = Path(masks) if masks else mask_cache(height, width)
    if masks.name.startswith('GCDID_'):
        return masks.with_name('GCIDC_flow_' + masks.name[len('GCDID_'):])
    return masks.with_name(masks.name + '_flow')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('height', type=int)
    parser.add_argument('width', type=int)
    parser.add_argument('split', nargs='?')
    parser.add_argument('--flow-for', type=Path)
    args = parser.parse_args()
    if args.flow_for:
        print(flow_cache(args.height, args.width, args.flow_for))
    else:
        print(mask_cache(args.height, args.width, args.split))
