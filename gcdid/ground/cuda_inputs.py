"""Load the RGB-derived semantic road mask and the camera convention used by the trainer."""
from pathlib import Path
import numpy as np
import torch
from .settings import H, W, ROAD
from ..sampling import nearest_axis

def intrinsics(height,width,device):
    return torch.tensor([[.58*width,0,.5*width],[0,1.92*height,.5*height],[0,0,1]],device=device,dtype=torch.float64)

def optical_matrix(data,key,width,height,device):
    from .camera import correspondence_intrinsics
    fx,fy,cx,cy=correspondence_intrinsics(data,key,W,H,width,height,device).unbind()
    matrix=torch.eye(3,device=device,dtype=torch.float64)
    matrix[0,0],matrix[1,1],matrix[0,2],matrix[1,2]=fx,fy,cx,cy
    return matrix


def inputs(roads,key,device,data):

    with np.load(Path(roads)/(key+'.npz'),allow_pickle=True) as z:
        width,height=int(z['image_width']),int(z['image_height'])
        ids=[i for i,name in enumerate(z['labels']) if str(name).lower() in ROAD]
        if not ids:
            road=torch.zeros((H,W),device=device,dtype=torch.bool)
        else:
            masks=torch.as_tensor(np.ascontiguousarray(z['masks'][ids]),device=device)
            y=nearest_axis(int(z['image_height']),H,device)
            x=nearest_axis(int(z['image_width']),W,device)
            road=masks[:,y[:,None],x].bool().any(0)
    return road,optical_matrix(data,key,width,height,device)

