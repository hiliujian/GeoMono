"""Read rectified camera optics for the GCIDC correspondence grid."""
import json
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=None)
def optics(folder, camera):
    table = folder / 'camera_intrinsics.json'
    if table.is_file():
        return json.loads(table.read_text())[camera]
    name = 'P_rect_' + camera
    for line in (folder / 'calib_cam_to_cam.txt').read_text().splitlines():
        key, sep, numbers = line.partition(':')
        if sep and key == name:
            projection = list(map(float, numbers.split()))
            return [projection[0], projection[5], projection[2], projection[6]]
    raise ValueError('Missing rectified camera intrinsics for ' + camera)


def correspondence_intrinsics(data, key, width, height, native_width, native_height, device):
    date, _, _, side = key.split('__')
    camera = '02' if side in ('l', '2') else '03'
    q = torch.tensor(optics(Path(data) / date, camera), device=device, dtype=torch.float64)
    q[2:] += .5
    q *= q.new_tensor((width / native_width, height / native_height,
                       width / native_width, height / native_height))
    q[2:] -= .5
    return q
