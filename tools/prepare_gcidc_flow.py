"""Verify GCIDC correspondence coverage and fill any missing selected frames."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcdid.correspondence import complete_flow, correspondence_cache
from gcdid.temporal import DETECTOR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "gcdid", "cache"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.gcdid / "config.json").read_text())
    if config.get("detector") != DETECTOR or config.get("geometry_backend") != "cuda":
        raise ValueError("Expected a GCDID mask cache: " + str(args.gcdid))
    with correspondence_cache(args.cache, args.data, config["height"], config["width"]) as (frames, spec):
        complete_flow(args.gcdid, frames, spec)


if __name__ == "__main__":
    main()
