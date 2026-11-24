"""Shared calibrated detector grid, road classes and KITTI file format."""
from functools import lru_cache
import json
from pathlib import Path

CALIBRATION = Path(__file__).with_name("calibration.json")
H, W = 192, 640
ROAD = {"road", "road marking"}
@lru_cache(maxsize=1)
def motion_calibration():
    return json.loads(CALIBRATION.read_bytes())
