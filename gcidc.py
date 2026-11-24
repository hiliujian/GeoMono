"""Single-frame ground-contact inverse-depth supervision on CUDA."""
import math

import torch
from torch.nn import functional as F

from gcdid.ground.cuda_ops import envelopes

EPS = 1e-8


def masked_median(values, valid):
    """Match torch.median's lower middle element without compacting on the host."""
    result = torch.nanmedian(values.masked_fill(~valid, float('nan')),dim=-1).values
    return torch.nan_to_num(result,nan=0.)


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


def image_rays(depth, K):
    height, width = depth.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height, device=depth.device, dtype=depth.dtype),
                           torch.arange(width, device=depth.device, dtype=depth.dtype), indexing='ij')
    rx = (x[None] - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None]
    ry = (y[None] - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None]
    return torch.stack((rx, ry, torch.ones_like(rx)), -1)


def road_plane(depth, background, rays):
    """Fit a road plane with a vertical initialization and measured residuals."""
    rays = rays.flatten(1, 2)
    support = (background & torch.isfinite(depth) & (depth > 0)).flatten(1) & (rays[..., 1] > 0)
    points = rays * torch.where(support, depth.flatten(1), 0.)[..., None]
    height = points[..., 1]
    initial_error = height - masked_median(height, support)[:, None]

    def fit(residual):
        spread = masked_median(residual.abs(), support).clamp_min(EPS)
        weights = support / (1 + (residual / spread[:, None]).square())
        mass = weights.sum(1, keepdim=True).clamp_min(EPS)
        deviation = points - (points * weights[..., None]).sum(1)[:, None] / mass[..., None]
        covariance = (deviation * weights[..., None]).transpose(1, 2) @ deviation
        normal = smallest_vector(covariance)
        normal = torch.where(normal[:, 1, None] < 0, -normal, normal)
        heights = (points * normal[:, None]).sum(2)
        core = support & (residual.abs() <= spread[:, None])
        offset = masked_median(heights, core)
        return normal, offset, heights - offset[:, None], covariance

    _, _, residual, _ = fit(initial_error)
    normal, offset, residual, covariance = fit(residual)
    eig = torch.linalg.eigvalsh(covariance.double())
    valid = (offset > 0) & (normal[:, 1] > 0)
    valid &= eig[:, 1] > torch.finfo(depth.dtype).eps * eig[:, 2]
    return valid, normal, offset, residual.view_as(depth)


def measure_contacts(depth, grid, road, K, images, x, bottom, outline, below, adjacent, selected):
    """Measure inverse depth at visible contacts and along the lower contour."""
    rays = image_rays(depth, K)
    background = grid < 0
    ground = road.bool() & background
    valid, normal, offset, residual = road_plane(depth, background, rays)
    support = ground & (rays[..., 1] > 0) & torch.isfinite(residual)
    spread = masked_median(residual.abs().flatten(1), support.flatten(1))
    plane = (rays * normal[:, None, None]).sum(-1) / offset[:, None, None].clamp_min(EPS)
    k = K[images]
    nx, ny, nz = normal[images].unbind(-1)
    instance_offset = offset[images, None].clamp_min(EPS)
    step = ny[:, None] / (k[:, 1, 1, None] * instance_offset)
    foot = plane.flatten()[below] - step
    observed = adjacent.clamp_min(EPS).reciprocal()
    error = (observed - foot - step).abs()
    uncertainty = observed * (spread[images, None] / instance_offset) + step.abs()
    road_below = ground.flatten()[below]
    contact = selected[:, None] & (bottom >= 0) & (bottom < depth.shape[1] - 1)
    contact &= background.flatten()[below] & torch.isfinite(adjacent) & (adjacent > 0)
    known = contact & valid[images, None] & (foot > 0) & (step > 0)
    known &= road_below | (error <= uncertainty)
    weight = 1 / (1 + (error / uncertainty.clamp_min(EPS)).square())
    centre = (foot + observed * weight) / (1 + weight)
    rx = (x.to(depth.dtype) - k[:, 0, 2, None]) / k[:, 0, 0, None]
    ry = (outline.to(depth.dtype) + .5 - k[:, 1, 2, None]) / k[:, 1, 1, None]
    contour = (nx[:, None] * rx + ny[:, None] * ry + nz[:, None]) / instance_offset
    return known, centre, contour, road_below


def local_profile(centre, known, x, samples):
    """Smooth measured contacts with their observed column distribution."""
    width = known.shape[1]
    columns = torch.where(known, x, width).sort(1).values[:, :samples]
    valid = columns < width
    indices = columns.clamp_max(width - 1)
    points = indices.to(centre.dtype)
    values = torch.where(valid, centre.gather(1, indices), 0.)
    count = known.sum(1, keepdim=True).to(centre.dtype)
    mean = (points * valid).sum(1, keepdim=True) / count
    variance = ((points - mean).square() * valid).sum(1, keepdim=True) / count
    # The variance of uniform subpixel quantization is 1/12 pixel squared.
    bandwidth = (variance + 1. / 12.).sqrt() * (4. / (3. * count)).pow(1. / 5.)
    first = torch.where(known, x, width).amin(1, keepdim=True)
    last = torch.where(known, x, 0).amax(1, keepdim=True)
    query = x.clamp(first, last).to(centre.dtype)
    delta = points[:, None] - query[..., None]
    log_weights = (-0.5 * (delta / bandwidth[..., None]).square()).masked_fill(~valid[:, None], -torch.inf)
    weights = (log_weights - log_weights.amax(-1, keepdim=True)).exp()
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(EPS)
    mean_x = (weights * delta).sum(-1)
    centred = delta - mean_x[..., None]
    var_x = (weights * centred.square()).sum(-1)
    mean_depth = (weights * values[:, None]).sum(-1)
    slope = (weights * centred * (values[:, None] - mean_depth[..., None])).sum(-1) / var_x.clamp_min(EPS)
    fit = mean_depth - slope * mean_x
    return torch.where((fit > 0) & (var_x > 0), fit, mean_depth)


def contact_profile(centre, known, x, batch, height):
    count = known.sum(1)
    ids = torch.where(count > 0)[0]
    ids = ids[count[ids].argsort(descending=True)]
    result = torch.zeros_like(centre)
    start = 0
    while start < ids.numel():
        samples = int(count[ids[start]])
        length = max(1, batch * height // samples)
        selected = ids[start:start + length]
        result[selected] = local_profile(centre[selected], known[selected], x, samples)
        start += length
    return result


def extend_facet(profile, contour, outline, bottom, facets, x, grounded):
    """Extend the most supported contour segment under a lifted overhang."""
    occupied = bottom >= 0
    width = bottom.shape[1]
    relief = (outline.to(profile.dtype) - bottom).clamp_min(0.) * occupied
    area = torch.zeros_like(profile).scatter_add_(1, facets.long(), relief)
    largest, support = area.max(1, keepdim=True)
    supported = (facets == support) & occupied
    left = torch.where(supported, x, width).amin(1, keepdim=True).clamp_max(width - 1)
    right = torch.where(supported, x, -1).amax(1, keepdim=True).clamp_min(0)
    span = right - left
    fraction = (x - left).to(profile.dtype) / span.clamp_min(1)
    extension = torch.lerp(contour.gather(1, left), contour.gather(1, right), fraction)
    distance = torch.maximum(left - x, x - right).clamp_min(0)
    gain = span / (span + distance).clamp_min(1)
    use = grounded & (largest > 0) & (span > 0) & (distance > 0)
    use &= (extension > 0) & torch.isfinite(extension)
    return torch.where(use, torch.lerp(profile, torch.maximum(profile, extension), gain), profile)


def extend_crop(profile, contour, bottom, known, road_below, x, height):
    """Continue the ground slope across the image's bottom boundary."""
    width = bottom.shape[1]
    x = x.expand_as(bottom)
    contact_x = masked_median(x.to(profile.dtype), known).long()[:, None]
    contact = contour.gather(1, contact_x)
    # The last two row centres can represent clipping within a pixel footprint.
    clipped = (bottom >= height - 2) & ~road_below
    left = torch.where(clipped & (x < contact_x), x, -1).amax(1, keepdim=True)
    right = torch.where(clipped & (x > contact_x), x, width).amin(1, keepdim=True)
    use_left = (left >= 0) & (x < left)
    use_right = (right < width) & (x > right)
    edge = torch.where(use_left, left, right).clamp(0, width - 1)
    entry = contour.gather(1, edge)
    slope = (entry - contact) / (edge - contact_x).abs().clamp_min(1)
    continuation = entry + slope * (x - edge).abs()
    use = (use_left | use_right) & known.any(1, keepdim=True) & (slope > 0)
    return torch.where(use & torch.isfinite(continuation), torch.maximum(profile, continuation), profile)


@torch.no_grad()
def build_targets(depth, grid, moving, K, cells, cell_valid, batch_index, road):
    """Write one inverse-depth target for every visible pixel of each grounded instance."""
    if not depth.is_cuda:
        raise ValueError('GCIDC target generation requires CUDA tensors')
    batch, height, width = depth.shape
    if not cell_valid.any():
        empty = torch.zeros_like(moving, dtype=torch.bool)
        return dict(u_t=torch.zeros_like(depth), mask=empty, original_mask=empty)
    area = height * width
    index = cells + batch_index[:, None] * area
    raw = depth.flatten()[index]
    valid = cell_valid & torch.isfinite(raw) & (raw > 0)
    selected = (valid & moving.flatten()[index]).any(1)
    rows, cols = cells.div(width, rounding_mode='floor'), cells.remainder(width)
    bottom = cells.new_full((len(cells), width), -1)
    bottom.scatter_reduce_(1, cols, torch.where(valid, rows, -1), reduce='amax', include_self=True)
    x = torch.arange(width, device=depth.device)[None]
    below = batch_index[:, None] * area + (bottom + 1).clamp(max=height-1) * width + x
    adjacent = depth.flatten()[below]

    outline, facets = envelopes(bottom)
    known, centre, contour, road_below = measure_contacts(
        depth, grid, road, K, batch_index, x, bottom, outline, below, adjacent, selected)
    grounded = known.any(1, keepdim=True)
    local = contact_profile(centre, known, x, batch, height)
    farthest = torch.where(known, centre, torch.inf).amin(1, keepdim=True)
    profile = torch.where(grounded, torch.maximum(local, torch.maximum(contour, farthest)), 0.)
    profile = extend_facet(profile, contour, outline, bottom, facets, x, grounded)
    profile = extend_crop(profile, contour, bottom, known, road_below, x, height)

    values = profile.gather(1, cols)
    used = valid & grounded & moving.flatten()[index]
    pixels = batch * area
    slots = torch.where(used, index, pixels).flatten()
    target = depth.new_zeros(pixels + 1)
    target.scatter_(0, slots, values.flatten())
    target = target[:-1].view_as(depth)
    anchors = known.gather(1, cols) & (rows == bottom.gather(1, cols))
    original = torch.zeros(pixels + 1, device=depth.device, dtype=torch.bool)
    original.scatter_(0, slots, anchors.flatten())
    return dict(u_t=target, mask=target > 0, original_mask=original[:-1].view_as(depth))


def loss_per_image(depth, target, mask):
    """Average relative inverse-depth error over each image's supervised pixels."""
    inverse = torch.where(mask, depth, 1.).clamp_min(EPS).reciprocal()
    target = torch.where(mask, target.detach(), 1.).clamp_min(EPS)
    relative = (inverse - target).abs() / target
    return relative.sum((-1, -2)) / mask.sum((-1, -2)).clamp_min(1)
