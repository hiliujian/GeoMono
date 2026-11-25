"""Load the RGB-derived semantic road mask and the camera convention used by the trainer."""
from pathlib import Path
import numpy as np
import torch
from .settings import H, W, ROAD
from .camera import intrinsics
from ..sampling import nearest_axis

def inputs(roads,key,device):

    with np.load(Path(roads)/(key+'.npz'),allow_pickle=True) as z:
        ids=[i for i,name in enumerate(z['labels']) if str(name).lower() in ROAD]
        if not ids:
            road=torch.zeros((H,W),device=device,dtype=torch.bool)
        else:
            masks=torch.as_tensor(np.ascontiguousarray(z['masks'][ids]),device=device)
            y=nearest_axis(int(z['image_height']),H,device)
            x=nearest_axis(int(z['image_width']),W,device)
            road=masks[:,y[:,None],x].bool().any(0)
    return road,intrinsics(H,W,device)

