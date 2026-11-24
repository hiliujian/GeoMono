"""Ground Contact Inverse Depth Constraint (GCIDC).

The road fit and three-frame correspondences constrain inverse depth at visible
contacts. Within grounded instances, additional pixels use the contact profile
and its measured uncertainty to produce the same inverse-depth supervision.
"""
import math

import torch
from torch.nn import functional as F

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


def interpolate_contacts(profile):
    """Interpolate inverse depth inside the span of measured contacts."""
    width = profile.shape[1]
    cols = torch.arange(width, device=profile.device)[None].expand_as(profile)
    known = profile > 0
    left = torch.where(known, cols, -1).cummax(-1).values
    right = torch.where(known, cols, width).flip(-1).cummin(-1).values.flip(-1)
    bounded = (left >= 0) & (right < width)
    left = left.clamp(0, width - 1)
    right = right.clamp(0, width - 1)
    u0 = profile.gather(1, left).double()
    u1 = profile.gather(1, right).double()
    fraction = (cols - left).double() / (right - left).clamp_min(1)
    return u0 + (u1 - u0) * fraction, bounded


def observed_cycle(flow, flipped):
    """Measure uncertainty from both neighbouring frame cycles."""
    batch, _, _, height, width = flow.shape
    dtype, device = flow.dtype, flow.device
    yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                            torch.arange(width, device=device, dtype=dtype), indexing='ij')
    pixel = torch.stack((xx, yy))
    limits = pixel.new_tensor((width - 1, height - 1))[:, None, None]
    start = pixel[None, None]
    dst = start + flow[:, ::2]
    finite = torch.isfinite(dst).all(2)
    inside = finite & (dst >= 0).all(2) & (dst <= limits).all(2)
    dst = torch.where(inside[:, :, None], dst, start)
    grid = (2 * dst / limits - 1).flatten(0, 1).permute(0, 2, 3, 1)
    reverse = F.grid_sample(flow[:, 1::2].flatten(0, 1), grid, mode='bilinear', align_corners=True)
    back = dst + reverse.reshape(batch, 2, 2, height, width)
    returned = torch.isfinite(back).all(2) & (back >= 0).all(2) & (back <= limits).all(2)
    observed = (inside & returned).all(1)
    extent = (back - start).abs().amax(1)
    extent = torch.where(observed[:, None], extent, 0.)
    return (torch.where(flipped[:, None, None], observed.flip(-1), observed),
            torch.where(flipped[:, None, None, None], extent.flip(-1), extent))


def measured_plane(depth, roadway, K):
    """Fit a plane, then refit with a Cauchy scale measured from its residuals."""
    batch, height, width = depth.shape
    y, x = torch.meshgrid(torch.arange(height, device=depth.device, dtype=depth.dtype),
                           torch.arange(width, device=depth.device, dtype=depth.dtype), indexing='ij')
    rx = (x[None] - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None]
    ry = (y[None] - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None]
    rays = torch.stack((rx, ry, torch.ones_like(rx)), 1).flatten(2)
    valid = (roadway & torch.isfinite(depth) & (depth > 0) & (ry > 0)).flatten(1)
    z = torch.where(valid, depth.flatten(1), 0.)
    points = rays * z[:, None]

    def weighted_plane(weights):
        total = weights.sum(1).clamp_min(EPS)
        centre = (points * weights[:, None]).sum(2) / total[:, None]
        delta = points - centre[:, :, None]
        covariance = (delta * weights[:, None]) @ delta.transpose(1,2)
        n = smallest_vector(covariance)
        position = (n[:, :, None] * points).sum(1)
        offset = masked_median(position, valid)
        sign = torch.where(offset < 0, -1., 1.)
        return n * sign[:, None], offset.abs(), position * sign[:, None], covariance

    _, h, ge, _ = weighted_plane(valid.to(depth.dtype))
    residual = ge - h[:, None]
    # The absolute median residual gives the scale of a centred Cauchy model.
    spread = masked_median(residual.abs(), valid).clamp_min(EPS)
    weights = valid / (1 + (residual / spread[:, None]).square())
    n, h, pos, covariance = weighted_plane(weights)
    sigma = masked_median((pos - h[:, None]).abs(), valid)
    eig = torch.linalg.eigvalsh(covariance.double())
    rank = (eig[:, 1] > torch.finfo(depth.dtype).eps * eig[:, 2]) & (h > 0)
    ray_product = (n[:, :, None] * rays).sum(1).view(batch,height,width)
    plane = torch.where(ray_product>EPS,h[:,None,None]/ray_product.clamp_min(EPS),0.)
    return plane,rank,n,h,sigma


def contact_profile(anchor, lower, upper, bottom, normal, observed, x, K, samples):
    """Fit a local contact line and estimate both sources of inverse-depth error."""
    count = observed.sum(1, keepdim=True)
    width = observed.shape[1]
    columns = torch.where(observed, x, width).sort(1).values[:, :samples]
    valid = columns < width
    indices = columns.clamp_max(width - 1)
    points = indices.to(anchor.dtype)

    def gather(values):
        return torch.where(valid, values.gather(1, indices), 0.)

    z = gather(anchor)
    count = count.to(anchor.dtype)
    centre = (points * valid).sum(1, keepdim=True) / count
    variance = ((points - centre).square() * valid).sum(1, keepdim=True) / count
    # The variance of uniform subpixel quantization is 1/12 pixel squared.
    bandwidth = (variance + 1. / 12.).sqrt() * (4. / (3. * count)).pow(1. / 5.)
    first = torch.where(observed, x, width).amin(1, keepdim=True)
    last = torch.where(observed, x, 0).amax(1, keepdim=True)
    query = x.clamp(first, last).to(anchor.dtype)

    def weights_at(position, allowed):
        delta = points[:, None] - position[..., None]
        log_weights = (-0.5 * (delta / bandwidth[..., None]).square()).masked_fill(~allowed, -torch.inf)
        peak = torch.where(allowed.any(-1, keepdim=True), log_weights.amax(-1, keepdim=True), 0.)
        weights = (log_weights - peak).exp()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(EPS)
        mean_x = (weights * delta).sum(-1)
        centred = delta - mean_x[..., None]
        var_x = (weights * centred.square()).sum(-1)
        return weights, mean_x, centred, var_x

    def fit(values, weights, mean_x, centred, var_x):
        mean = (weights * values[:, None]).sum(-1)
        slope = (weights * centred * (values[:, None] - mean[..., None])).sum(-1) / var_x.clamp_min(EPS)
        return mean, slope, mean - slope * mean_x

    weights, mean_x, centred, var_x = weights_at(query, valid[:, None])
    mean, slope, estimate = fit(z, weights, mean_x, centred, var_x)
    linear = (estimate > 0) & (var_x > 0)
    estimate = torch.where(linear, estimate, mean)
    alpha = torch.where(linear[..., None],
                        weights - weights * centred * mean_x[..., None] / var_x.clamp_min(EPS)[..., None],
                        weights)
    # A negative fit coefficient exchanges the endpoints of its propagated interval.
    low_delta = alpha * (gather(lower) - z)[:, None]
    high_delta = alpha * (gather(upper) - z)[:, None]
    geometry_low = estimate + torch.minimum(low_delta, high_delta).sum(-1)
    geometry_high = estimate + torch.maximum(low_delta, high_delta).sum(-1)

    allowed = valid[:, :, None] & valid[:, None, :]
    allowed &= ~torch.eye(samples, device=anchor.device, dtype=torch.bool)
    loo = weights_at(points, allowed)
    _, _, held_out = fit(z, *loo)
    observable = valid & (count >= 3) & (loo[-1] > 0) & torch.isfinite(held_out)
    residual = torch.where(observable, (held_out - z).abs(), 0.)
    margin = (alpha.abs() * residual[:, None]).sum(-1)

    # A vertical surface normal is perpendicular to the estimated road normal.
    _, foot_slope, foot = fit(gather(bottom.to(anchor.dtype)), weights, mean_x, centred, var_x)
    normals = normal.gather(1, indices[..., None].expand(-1, -1, 3))
    nx, ny, nz = (weights @ normals).unbind(-1)
    fx, fy = K[:, 0, 0, None], K[:, 1, 1, None]
    cx, cy = K[:, 0, 2, None], K[:, 1, 2, None]
    u = nx * fx + nz * (cx - query)
    v = ny * fy + nz * (cy - foot)
    denominator = v - u * foot_slope
    divided = denominator.abs() > EPS
    slope_y = -(u * slope + nz * estimate) / torch.where(divided, denominator, 1.)
    slope_y = torch.where(linear & divided & torch.isfinite(slope_y), slope_y, 0.)
    return estimate, geometry_low, geometry_high, margin, foot, slope_y, first, last


@torch.no_grad()
def build_targets(depth, grid, moving, K, cells, cell_valid, batch_index, road, flow, intrinsics, poses, flipped,
                  anchor_mask=None):
    """Build one inverse-depth target map from grounded-instance evidence."""
    if not depth.is_cuda:
        raise ValueError('GCIDC target generation requires CUDA tensors')
    batch,height,width = depth.shape
    if flow.shape != (batch, 4, 2, height, width):
        raise ValueError('GCIDC needs both directions for each neighbouring frame')
    instances = cells.shape[0]
    roadway = road.bool() & (grid < 0)
    _,road_valid,road_normal,road_offset,road_sigma = measured_plane(depth, roadway, K)
    plane,plane_valid,normal,offset,sigma = measured_plane(depth, grid < 0, K)
    tracks,extent = observed_cycle(flow, flipped)
    area = height * width
    flat = depth.flatten()
    index = cells + batch_index[:, None] * area
    raw = flat[index]
    valid = cell_valid & torch.isfinite(raw) & (raw > 0)
    selected = (valid & moving.flatten()[index]).any(1)
    rows,cols = cells.div(width, rounding_mode='floor'),cells.remainder(width)
    bottom = torch.full((instances,width),-1,device=depth.device,dtype=torch.long)
    anchor = valid if anchor_mask is None else valid & anchor_mask.flatten()[index]
    bottom.scatter_reduce_(1,cols,torch.where(anchor,rows,-1),reduce='amax',include_self=True)
    x = torch.arange(width, device=depth.device)[None]
    below = batch_index[:, None] * area + (bottom+1).clamp(max=height-1)*width + x
    adjacent = flat[below]
    contact = selected[:, None] & (bottom >= 0) & (bottom < height - 1)
    contact &= (grid.flatten()[below] < 0) & tracks.flatten()[below]
    contact &= torch.isfinite(adjacent) & (adjacent > 0)
    use_road = roadway.flatten()[below] & road_valid[batch_index, None]
    fit = torch.where(use_road, road_valid[batch_index, None], plane_valid[batch_index, None])
    h = torch.where(use_road, road_offset[batch_index, None], offset[batch_index, None])
    noise = torch.where(use_road, road_sigma[batch_index, None], sigma[batch_index, None])
    n = torch.where(use_road[..., None], road_normal[batch_index, None], normal[batch_index, None])
    contact &= fit & (h > noise)

    k = K[batch_index]
    nx,ny,nz = n.unbind(-1)
    rx = (x.to(depth.dtype)-k[:,0,2,None])/k[:,0,0,None]
    ry = (bottom.to(depth.dtype)-k[:,1,2,None])/k[:,1,1,None]
    foot = nx*rx + ny*ry + nz
    step = ny/k[:,1,1,None]
    reprojection = (nx.abs()*extent[:,0].flatten()[below]/k[:,0,0,None] +
                    ny.abs()*extent[:,1].flatten()[below]/k[:,1,1,None])
    observed = adjacent.clamp_min(EPS).reciprocal()
    offset_inverse = step/h.clamp_min(EPS)
    flow_uncertainty = reprojection/h.clamp_min(EPS)
    measured_low = torch.minimum(observed, observed-offset_inverse) - flow_uncertainty
    measured_high = torch.maximum(observed, observed-offset_inverse) + flow_uncertainty
    plane_low = (torch.minimum(foot,foot+step)-reprojection)/(h+noise).clamp_min(EPS)
    plane_high = (torch.maximum(foot,foot+step)+reprojection)/(h-noise).clamp_min(EPS)
    low = torch.maximum(measured_low, plane_low)
    high = torch.minimum(measured_high, plane_high)
    trusted = contact & (low > 0) & (high >= low) & torch.isfinite(low) & torch.isfinite(high)
    low_profile,bounded = interpolate_contacts(torch.where(trusted,low,0.))
    high_profile,_ = interpolate_contacts(torch.where(trusted,high,0.))
    inverse = raw.clamp_min(EPS).reciprocal()
    candidate = inverse.double().clamp(low_profile.gather(1,cols),high_profile.gather(1,cols)).to(inverse.dtype)
    direct = valid & bounded.gather(1,cols) & (candidate != inverse)
    applied = direct.clone()

    plane_contact = (foot + 0.5 * step) / h.clamp_min(EPS)
    midpoint = (foot / h.clamp_min(EPS) + observed) * 0.5
    profile = (trusted | (contact & use_road & (plane_contact > 0) & torch.isfinite(plane_contact)))
    profile &= trusted.any(1, keepdim=True) & (midpoint > 0) & torch.isfinite(midpoint)
    counts = profile.sum(1)
    profiles = torch.where(counts > 0)[0]
    order = profiles[counts[profiles].argsort(descending=True)]
    lower = torch.minimum(midpoint, (measured_low + plane_low) * 0.5)
    upper = torch.maximum(midpoint, (measured_high + plane_high) * 0.5)
    start = 0
    while start < order.numel():
        samples = int(counts[order[start]])
        # Bound the temporary profile tensor by the size of the image batch.
        length = max(1, batch * height // samples)
        ids = order[start:start + length]
        start += length
        estimate, geo_lo, geo_hi, margin, foot_row, slope_y, first, last = contact_profile(
            midpoint[ids], lower[ids], upper[ids], bottom[ids], n[ids], profile[ids], x, k[ids], samples)
        cols_i = cols[ids]
        raw_i = inverse[ids]
        estimate_i = estimate.gather(1, cols_i)
        upright = estimate_i + slope_y.gather(1, cols_i) * (rows[ids] - foot_row.gather(1, cols_i))
        upright = torch.where((upright > 0) & torch.isfinite(upright), upright, estimate_i)
        geometry = raw_i.clamp(geo_lo.gather(1, cols_i), geo_hi.gather(1, cols_i))
        error = margin.gather(1, cols_i)
        held_out = raw_i.clamp(estimate_i - error, estimate_i + error)

        low_i, high_i = torch.minimum(raw_i, upright), torch.maximum(raw_i, upright)
        geometry = torch.where((geometry > 0) & torch.isfinite(geometry), geometry, raw_i)
        held_out = torch.where((held_out > 0) & torch.isfinite(held_out), held_out, raw_i)
        unsupported_geometry = (upright - geometry.clamp(low_i, high_i)).abs()
        unsupported_fit = (upright - held_out.clamp(low_i, high_i)).abs()
        correction = upright - raw_i
        change = correction.abs()
        amount = change / (change + unsupported_geometry + unsupported_fit).clamp_min(torch.finfo(raw_i.dtype).tiny)
        extended = raw_i + amount * correction

        supported = ((cols_i >= first) & (cols_i <= last)) | tracks.flatten()[index[ids]]
        extend = (valid[ids] & ~direct[ids] & supported & (upright > 0) & (extended > 0)
                  & torch.isfinite(extended) & (extended != raw_i))
        candidate[ids] = torch.where(extend, extended, candidate[ids])
        applied[ids] = direct[ids] | extend

    target_index = torch.where(applied,index,batch*area).flatten()

    def raster(values):
        output = values.new_zeros(batch*area+1)
        output.scatter_(0,target_index,values.flatten())
        return output[:-1].view(batch,height,width)

    target,mask = raster(candidate),raster(applied)
    original = raster(direct)
    return dict(u_t=target,mask=mask,original_u_t=torch.where(original, target, 0.),original_mask=original,
                n=normal,c0=offset,sigma=sigma,z_plane=plane)


def loss_per_image(depth,target,mask):
    """Relative inverse-depth loss; target generation is detached."""
    inverse = depth.clamp_min(EPS).reciprocal()
    relative = (inverse-target.detach()).abs()/target.detach().clamp_min(EPS)
    return torch.where(mask,relative,torch.zeros_like(relative)).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1)

