"""Load packed instance data and rasterize ownership grids on CUDA."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from gcdid.sampling import nearest_axis

INDEX_TO_CLASS=("car","truck","bus","person","rider","bicycle","motorcycle","train","tram")
GROUPS={
    'car':0,'van':0,'truck':0,'bus':0,'train':0,'tram':0,'vehicle':0,
    'car_(van)':0,'truck_(truck)':0,
    'person':1,'pedestrian':1,
    'rider':2,'bicycle':2,'motorcycle':2,'motorbike':2,'cyclist':2,
    'bicyclist':2,'motorcyclist':2,
}


def frame_key(line):
    parts=line.strip().split();date,drive=parts[0].split('/')[:2]
    return f'{date}__{drive}__{int(parts[1]):010d}__{parts[2]}'


def at_frame(key,number):
    date,drive,_,side=key.split('__')
    return f'{date}__{drive}__{number:010d}__{side}'


def image_path(data,key,extension='.jpg'):
    date,drive,number,side=key.split('__')
    camera='image_02' if side in ['l','2'] else 'image_03'
    return Path(data)/date/drive/camera/'data'/(number+extension)


def candidate_names(labels):
    names = (str(name).strip().lower() for name in labels)
    return [INDEX_TO_CLASS[int(name)] if name.isdigit() and int(name) < len(INDEX_TO_CLASS)
            else name for name in names]


def candidate_grids(path, height, width):
    """Decode packed crop bits directly at the two training orientations on CUDA."""
    path = Path(path)
    empty = torch.full((height,width), -1, device="cuda", dtype=torch.int16)
    with np.load(path,allow_pickle=False) as z:
        names = candidate_names(z["labels"])
        groups = np.asarray([GROUPS[s] for s in names],np.int8)
        if not len(groups):
            return empty, empty, groups
        nh,nw=int(z["height"]),int(z["width"])
        boxes=torch.as_tensor(z["boxes"],device="cuda",dtype=torch.long)
        offsets=torch.as_tensor(z["masks_off"][:len(groups)],device="cuda",dtype=torch.long)
        bits=torch.as_tensor(z["masks_bits"],device="cuda",dtype=torch.uint8)
    if bits.numel()==0:
        return empty, empty, groups
    sx=nearest_axis(nw,width,empty.device)
    sy=nearest_axis(nh,height,empty.device)
    x=torch.cat((sx,nw-1-sx))[None,None,:]
    y=sy[None,:,None]
    grid=empty.repeat(1,2)
    # Process at most eight candidates at once at 1024x320. The first instance
    # in candidate order owns overlaps, identically for both orientations.
    for i in range(0,len(groups),8):
        x0,y0,x1,y1=boxes[i:i+8,:,None,None].unbind(1)
        covered=(x>=x0)&(x<=x1)&(y>=y0)&(y<=y1)
        bit=(y-y0)*(x1-x0+1)+x-x0
        block=offsets[i:i+8,None,None]+bit.div(8,rounding_mode="floor")
        hit=covered & (((bits[block.clamp(0,bits.numel()-1)] >> (7-bit.remainder(8)))&1).bool())
        idx=hit.byte().argmax(0).to(torch.int16)+i
        grid=torch.where((grid<0)&hit.any(0),idx,grid)
    return grid[:,:width].contiguous(),grid[:,width:].contiguous(),groups


def region_cells(grid,number):
    """Group grid pixels on CUDA for GCIDC target construction during training."""
    flat=grid.flatten()
    idx=torch.where(flat>=0)[0]
    owners=flat[idx].long()
    order=torch.argsort(owners,stable=True)
    counts=torch.bincount(owners,minlength=number)
    return idx[order].int(),F.pad(counts.cumsum(0),(1,0)).int()
