"""mmseg.models.builder stand-in: MPViT only decorates with BACKBONES."""
from mmcv.utils import Registry

BACKBONES = Registry("backbone")
