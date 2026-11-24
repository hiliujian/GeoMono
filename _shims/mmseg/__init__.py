"""Minimal stand-in for mmseg.

MonoViT's networks/mpvit.py only needs two things from mmseg:
  * mmseg.utils.get_root_logger
  * mmseg.models.builder.BACKBONES

Both are thin wrappers over mmcv, which is installed, so this package keeps the
imports working without pulling in the full mmsegmentation stack (whose pinned
versions conflict with the torch/mmcv pair used here).  No numerics live here.
"""
