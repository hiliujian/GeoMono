"""Ground Contact Inverse Depth Constraint (GCIDC).

Contact geometry fixes observed inverse depths. Three-frame correspondences
constrain unobserved vehicle columns in the same target. The detached target
and its validity mask supervise one relative inverse-depth loss on CUDA.
"""
import math
import torch
import torch.nn.functional as F
# Constants for robust ground fitting and spatial column smoothing.
C_ROBUST = 2.0
ITERS = 5
EPS = 1e-8
MAX_FACTOR = 3.0
SMOOTH_COLS = 16


def masked_median(values, valid):
    """Match torch.median's lower middle element without compacting on the host."""
    result = torch.nanmedian(values.masked_fill(~valid, float('nan')),dim=-1).values
    return torch.nan_to_num(result,nan=0.)


def masked_mad(values, valid):
    centre = masked_median(values,valid)
    return 1.4826*masked_median((values-centre[...,None]).abs(),valid).clamp_min(EPS)


def smallest_vector(covariance):
    """Closed-form symmetric 3x3 eigensystem on CUDA, with stable FP64 scalars.

    A cross product of two rows of C-lambda_min*I gives the smallest eigenvector.
    The largest of the three cross products avoids a vanishing row pair. FP64
    is restricted to these tiny matrices to retain the small ground eigenvalue.
    """
    c = covariance.double()
    eye = torch.eye(3,device=c.device,dtype=c.dtype)
    q = c.diagonal(dim1=-2,dim2=-1).sum(-1)/3.
    a = c-q[...,None,None]*eye
    p = (a.square().sum((-1,-2))/6.).sqrt().clamp_min(torch.finfo(c.dtype).tiny)
    b = a/p[...,None,None]
    determinant = (b[...,0,0]*(b[...,1,1]*b[...,2,2]-b[...,1,2]*b[...,2,1])
                   -b[...,0,1]*(b[...,1,0]*b[...,2,2]-b[...,1,2]*b[...,2,0])
                   +b[...,0,2]*(b[...,1,0]*b[...,2,1]-b[...,1,1]*b[...,2,0]))
    phi = (determinant*.5).clamp(-1.,1.).acos()/3.
    eigenvalue = q+2.*p*torch.cos(phi+2.*math.pi/3.)
    shifted = c-eigenvalue[...,None,None]*eye
    cross = torch.stack((torch.cross(shifted[...,0,:],shifted[...,1,:],dim=-1),
                         torch.cross(shifted[...,0,:],shifted[...,2,:],dim=-1),
                         torch.cross(shifted[...,1,:],shifted[...,2,:],dim=-1)),dim=-2)
    lengths = cross.square().sum(-1)
    choice = lengths.argmax(-1)
    vector = cross.gather(-2,choice[...,None,None].expand(*choice.shape,1,3)).squeeze(-2)
    # A repeated eigenvalue permits any vector in its nullspace. Use a row-
    # orthogonal vector; a zero covariance receives a finite coordinate axis.
    row_index = shifted.square().sum(-1).argmax(-1)
    row = shifted.gather(-2,row_index[...,None,None].expand(*row_index.shape,1,3)).squeeze(-2)
    axis = F.one_hot(row.abs().argmin(-1),3).to(row.dtype)
    fallback = torch.cross(row,axis,dim=-1)
    fallback = torch.where((fallback.square().sum(-1)>0)[...,None],fallback,axis)
    vector = torch.where((lengths.amax(-1)>0)[...,None],vector,fallback)
    vector = vector/vector.norm(dim=-1,keepdim=True).clamp_min(torch.finfo(c.dtype).tiny)
    return vector.to(covariance.dtype)


def ground_geometry(depth,seed_mask,K):
    """Fit one ground plane per image with batched IRLS."""
    B,H,W = depth.shape
    yy,xx = torch.meshgrid(torch.arange(H,device=depth.device,dtype=depth.dtype),
                          torch.arange(W,device=depth.device,dtype=depth.dtype),indexing='ij')
    rx = (xx[None]-K[:,0,2,None,None])/K[:,0,0,None,None]
    ry = (yy[None]-K[:,1,2,None,None])/K[:,1,1,None,None]
    rays = torch.stack((rx,ry,torch.ones_like(rx)),1).flatten(2)
    seed = (seed_mask&(depth>0)&torch.isfinite(depth)&(ry>0)).flatten(1)
    z = torch.where(seed,depth.flatten(1),torch.zeros_like(depth.flatten(1)))
    points = rays*z[:,None]
    c0 = masked_median(points[:,1],seed)
    n = torch.stack((torch.zeros_like(c0),torch.ones_like(c0),torch.zeros_like(c0)),1)
    for _ in range(ITERS):
        res = (n[:,:,None]*points).sum(1)-c0[:,None]
        scale = masked_mad(res,seed)
        weights = seed/(1.+(res/(C_ROBUST*scale[:,None])).square())
        weight_sum = weights.sum(-1).clamp_min(EPS)
        mu = (points*weights[:,None]).sum(-1)/weight_sum[:,None]
        centred = points-mu[:,:,None]
        cov = (centred*weights[:,None])@centred.transpose(1,2)
        n = smallest_vector(cov)
        orient = (n[:,:,None]*points*weights[:,None]).sum((1,2))
        n = torch.where((orient<0)[:,None],-n,n)
        good = (weights>.5)&seed
        values = (n[:,:,None]*points).sum(1)
        c0 = torch.where(good.sum(-1)>64,masked_median(values,good),masked_median(values,seed))
    res = (n[:,:,None]*points).sum(1)-c0[:,None]
    good = (weights>.5)&seed
    sigma = torch.where(good.sum(-1)>64,masked_mad(res,good),masked_mad(res,seed))
    nr = (n[:,:,None]*rays).sum(1).view(B,H,W)
    plane = torch.where(nr>EPS,c0[:,None,None]/nr.clamp_min(EPS),torch.zeros_like(depth))
    on_ground = (depth*nr-c0[:,None,None]).abs()<C_ROBUST*sigma[:,None,None]
    valid = seed.sum(-1)>=256
    return plane,on_ground,valid,n,c0,sigma


def smooth_profile(raw, present, foot_prediction):
    """Smooth the inverse-depth profile along observed columns."""
    M, W = raw.shape
    cols = torch.arange(W, device=raw.device)[None].expand(M, W)
    compact = torch.where(present, cols, torch.full_like(cols, W)).sort(-1).values
    compact_valid = compact < W
    compact = compact.clamp(max=W-1)
    prof = raw.gather(1, compact) * compact_valid
    count = present.sum(-1)
    radius = (count-1).clamp(1, SMOOTH_COLS).to(raw.dtype)
    offsets = torch.arange(-SMOOTH_COLS, SMOOTH_COLS+1, device=raw.device, dtype=raw.dtype)
    kernel = torch.exp(-.5*(offsets[None]/(radius[:,None]/2.).clamp_min(.5)).square())
    kernel = kernel * (offsets.abs()[None] <= radius[:,None])
    kernel = kernel / kernel.sum(-1, keepdim=True)
    samples = F.pad(prof, (SMOOTH_COLS, SMOOTH_COLS)).unfold(-1, 2*SMOOTH_COLS+1, 1)
    observed = F.pad((prof>0).to(raw.dtype), (SMOOTH_COLS, SMOOTH_COLS)).unfold(-1, 2*SMOOTH_COLS+1, 1)
    numerator = (samples*kernel[:,None]).sum(-1)
    denominator = (observed*kernel[:,None]).sum(-1)
    smooth = numerator / denominator.clamp_min(1e-12)
    z = smooth.clamp_min(EPS).reciprocal()
    foot = foot_prediction.gather(1, compact)
    good = ((denominator>1e-6)&(smooth>0)&(foot>0)&
            (z<=MAX_FACTOR*foot)&(z*MAX_FACTOR>=foot))
    smooth = torch.where(good, smooth, torch.zeros_like(smooth))
    prof = torch.where((count>=3)[:,None], smooth, prof) * compact_valid
    ranks = (present.long().cumsum(-1)-1).clamp_min(0)
    return prof.gather(1, ranks) * present


def extend_profile(profile):
    """Harmonic gap interpolation and zero-slope end extension of inverse depth.

    Known columns remain exact hard constraints. FP64 interpolation mirrors
    NumPy interp in the frozen reference; target storage remains FP32.
    """
    M, W = profile.shape
    cols = torch.arange(W, device=profile.device)[None].expand(M, W)
    known = profile > 0
    left = torch.where(known, cols, torch.full_like(cols, -1)).cummax(-1).values
    right = torch.where(known, cols, torch.full_like(cols, W)).flip(-1).cummin(-1).values.flip(-1)
    first = torch.where(known, cols, torch.full_like(cols, W)).amin(-1, keepdim=True)
    last = torch.where(known, cols, torch.full_like(cols, -1)).amax(-1, keepdim=True)
    left = torch.where(left>=0, left, first).clamp(0,W-1)
    right = torch.where(right<W, right, last).clamp(0,W-1)
    u0, u1 = profile.gather(1,left).double(), profile.gather(1,right).double()
    fraction = (cols-left).double() / (right-left).clamp_min(1).double()
    return u0+(u1-u0)*fraction


@torch.no_grad()
def build_targets(depth, grid, moving, K, cells, cell_valid, batch_index, road, flow, intrinsics, poses, flipped):
    """CUDA tensors: B,H,W images, B,3,3 K, padded M,L instance cell metadata.

    Road is the union of existing SAM road and road-marking masks, transformed
    in the same orientation as the target RGB. It is not a motion mask.
    """
    if depth.device.type != 'cuda':
        raise ValueError('GCIDC target generation requires CUDA tensors')
    B,H,W = depth.shape
    M,L = cells.shape
    plane,ground,frame_valid,n,c0,sigma = ground_geometry(depth,grid<0,K)
    visible_road = road.bool() & (grid<0)
    road_plane,_,road_valid,road_n,road_h,_ = ground_geometry(depth,visible_road,K)
    flat = depth.reshape(-1)
    indices = cells + batch_index[:,None]*(H*W)
    values = flat[indices]
    valid = cell_valid & torch.isfinite(values) & (values>0)
    y,x = cells.div(W,rounding_mode='floor'),cells.remainder(W)
    count = valid.sum(-1)
    requested = (moving.flatten()[indices]&valid).any(-1)
    selected = requested & (count>=4)
    bottom = torch.full((M,W),-1,device=depth.device,dtype=torch.long)
    bottom.scatter_reduce_(1,x,torch.where(valid,y,torch.full_like(y,-1)),reduce='amax',include_self=True)
    present = bottom>=0
    cols = torch.arange(W,device=depth.device)[None].expand(M,W)
    below = (bottom+1).clamp(max=H-1)
    foot_idx = batch_index[:,None]*(H*W)+bottom.clamp_min(0)*W+cols
    below_idx = batch_index[:,None]*(H*W)+below*W+cols
    foot,zn = flat[foot_idx],flat[below_idx]

    def contact_profile(surface,on_ground):
        zc = surface.flatten()[foot_idx]
        ok = on_ground.flatten()[below_idx]&present&(bottom<H-1)&(surface.flatten()[below_idx]>0)
        bracket = ok&(zc>0)&(zn>0)
        zc = torch.where(bracket,2./(zc.clamp_min(EPS).reciprocal()+zn.clamp_min(EPS).reciprocal()),zc)
        keep = ok&(zc>0)&(zn>0)&(zc<=MAX_FACTOR*zn)&(zc*MAX_FACTOR>=zn)
        return torch.where(keep,zc.clamp_min(EPS).reciprocal(),torch.zeros_like(zc)),keep

    raw,_ = contact_profile(plane,ground)
    original_profile = smooth_profile(raw,present,foot)
    original_profile = original_profile*selected[:,None]*frame_valid[batch_index,None]
    old = original_profile.gather(1,x)
    original_valid = valid&(old>0)
    original_u = torch.where(original_valid,old,torch.zeros_like(old))
    has_original = original_valid.any(-1)

    # Verify road contact and evaluate the anchor at the bottom pixel boundary.
    _,contact_ok = contact_profile(road_plane,visible_road)
    bn,bh,bK = road_n[batch_index],road_h[batch_index],K[batch_index]
    rx = (cols.to(depth.dtype)-bK[:,0,2,None])/bK[:,0,0,None]
    ry = (bottom.clamp_min(0).to(depth.dtype)-bK[:,1,2,None])/bK[:,1,1,None]
    nr = bn[:,0,None]*rx+bn[:,1,None]*ry+bn[:,2,None]
    inverse_foot = (nr+bn[:,1,None]/bK[:,1,1,None]/2.)/bh[:,None]
    zc = inverse_foot.clamp_min(EPS).reciprocal()
    road_raw = torch.where(contact_ok,zc.clamp_min(EPS).reciprocal(),torch.zeros_like(zc))
    road_profile = smooth_profile(road_raw,present,foot)
    profile = torch.where(has_original[:,None],original_profile,road_profile)
    extended = extend_profile(profile).gather(1,x)
    z = extended.clamp_min(EPS).reciprocal()
    foot_at_cell = foot.gather(1,x)
    good = torch.isfinite(extended)&(extended>EPS)&(z<=MAX_FACTOR*foot_at_cell)&(z*MAX_FACTOR>=foot_at_cell)
    atomic = (good|~valid).all(-1) & (road_profile>0).any(-1)
    accepted = selected & road_valid[batch_index] & (has_original|atomic)
    added = valid&~original_valid&good&accepted[:,None]
    final_u = torch.where(added,extended.to(depth.dtype),original_u)
    final_valid = original_valid|added
    output_index = torch.where(valid,indices,torch.full_like(indices,B*H*W)).flatten()
    def raster(value):
        output = torch.zeros(B*H*W+1,device=depth.device,dtype=value.dtype)
        output.scatter_(0,output_index,value.flatten())
        return output[:-1].view(B,H,W)
    target, mask = raster(final_u), raster(final_valid)
    from gcidc_cuda import complete
    complete(depth, grid, cells, valid, batch_index, requested, road, target, mask,
             flow, intrinsics, poses, flipped)
    return dict(u_t=target,mask=mask,
                original_u_t=raster(original_u),original_mask=raster(original_valid),
                n=n,c0=c0,sigma=sigma,z_plane=plane)


def loss_per_image(depth,target,mask):
    """Relative inverse-depth loss; target generation is detached."""
    inverse = depth.clamp_min(EPS).reciprocal()
    relative = (inverse-target.detach()).abs()/target.detach().clamp_min(EPS)
    return torch.where(mask,relative,torch.zeros_like(relative)).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1)

