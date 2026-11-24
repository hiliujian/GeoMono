"""Official WAFT-a1 inference, used only during offline mask preparation."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
from PIL import Image
import torch
from gcdid.sampling import resize_rgb


class FlowModel:
    def __init__(self,root):
        root=Path(root)
        sys.path.insert(0,str(root/'third_party/WAFT'))
        from model.waft_a1 import WAFT

        config = root / 'third_party/WAFT/config/a1/tar-c-t.json'
        self.model = WAFT(SimpleNamespace(**json.loads(config.read_text()))).eval().cuda()
        weights=torch.load(root/'models/waft/tar-c-t.pth',map_location='cpu',weights_only=False)
        self.model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()})

    @staticmethod
    def image(path,height,width):
        with Image.open(path) as source:
            rgb=torch.from_numpy(np.array(source.convert('RGB'))).cuda()
        return resize_rgb(rgb,height,width)

    @torch.inference_mode()
    def pair(self,a,b):
        """Forward and reverse inference in one GPU batch."""
        return self.model(torch.stack((a, b)), torch.stack((b, a)))['flow'][-1]

    @torch.inference_mode()
    def triplet(self, paths, height, width, *, endpoints=False):
        before, center, after = (self.image(path, height, width) for path in paths)
        adjacent = torch.cat((self.pair(center, before), self.pair(center, after))).float()
        if endpoints:
            return adjacent, self.pair(before, after).float()
        return adjacent
