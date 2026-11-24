"""Batched GCDID on CUDA. Each row of the column tables describes one candidate."""
from functools import lru_cache
import torch
import torch.nn.functional as F
from .cuda_ops import envelopes
from .visual_pose import homogeneous, visual_scene
from .settings import H, W


@lru_cache(maxsize=8)
def pixels(device):
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    return torch.stack((x, y), -1)


def reverse_at(field, point):
    # Match the 5-bit bilinear interpolation table used to validate WAFT correspondences.
    ix, iy = (point.float() * 32).round().long().unbind(-1)
    x, y = ix >> 5, iy >> 5
    fx, fy = (ix % 32).float() / 32, (iy % 32).float() / 32
    coords = lambda yy, xx: field[yy.clamp(0, H - 1), xx.clamp(0, W - 1)]
    return (
        coords(y, x) * ((1 - fx) * (1 - fy))[..., None]
        + coords(y, x + 1) * (fx * (1 - fy))[..., None]
        + coords(y + 1, x) * ((1 - fx) * fy)[..., None]
        + coords(y + 1, x + 1) * (fx * fy)[..., None]
    )


def visible(xy):
    return (xy[..., 0] >= 0) & (xy[..., 0] <= W - 1) & (xy[..., 1] >= 0) & (xy[..., 1] <= H - 1)


class Scene:
    def __init__(self, road, K, poses, flow, normal, road_variance):
        self.K, self.poses = K, poses
        self.f = K.diag()[:2]
        self.c = K[:2, 2]
        xy = pixels(K.device)
        self.road_variance = road_variance
        obs = []
        par = []
        variance = []
        seen = []
        fields = flow.permute(0, 2, 3, 1).float()
        self.flow, self.reverse_flow = fields[::2], fields[1::2]
        for s, T in enumerate(poses):
            fw = self.flow[s]
            dst = xy.double() + fw
            back = reverse_at(self.reverse_flow[s], dst)
            cycle = (fw + back).square().sum(-1)
            image = dst.float().double()
            rays = homogeneous((image - self.c) / self.f) @ T[:3, :3]
            good = visible(image) & (rays[..., 2] > 0)
            q = rays[..., :2] / torch.where(good, rays[..., 2], 1.0)[..., None]
            b = T[:3, :3].T @ T[:3, 3]
            obs.append(q * self.f - (xy - self.c))
            par.append((b[:2] - q * b[2]) * self.f)
            variance.append(1.0 / 6 + cycle / 2)
            seen.append(good)
            if s == 0:
                self.road = road & torch.isfinite(cycle) & good
            else:
                self.road |= road & torch.isfinite(cycle) & good
        self.planes = normal.expand(2, -1)
        self.obs, self.par, self.var = (torch.stack(v) for v in (obs, par, variance))
        self.visible = seen[0] & seen[1]


def instance_median(values, owner, n):
    order = values.argsort(stable=True)
    sort = order[owner[order].argsort(stable=True)]
    counts = torch.bincount(owner, minlength=n)
    start = counts.cumsum(0) - counts
    v = F.pad(values[sort], (0, 1))
    return torch.where(
        counts > 0, (v[start + (counts - 1).clamp_min(0) // 2] + v[start + counts // 2]) * 0.5, 0.0
    )


class Instances:
    def __init__(self, grid, n):
        self.grid, self.n = grid.long(), n
        dev = grid.device
        self.x = torch.arange(W, device=dev).expand(n, -1)
        y, x = torch.where(self.grid >= 0)
        id = self.grid[y, x]
        self.size = n * W
        j = id * W + x

        def column_reduce(initial, reduction):
            v = torch.full((self.size,), initial, device=dev, dtype=torch.long)
            return v.scatter_reduce_(0, j, y, reduce=reduction).view(n, W)

        self.bottom = column_reduce(-1, "amax")
        self.top = column_reduce(H, "amin")
        self.count = torch.bincount(j, minlength=self.size).reshape(n, W)
        self.height = self.bottom.amax(1) - self.top.amin(1) + 1
        self.left = self.x.masked_fill(self.count == 0, W).amin(1)
        self.right = self.x.masked_fill(self.count == 0, -1).amax(1)
        self.hull, self.facets = envelopes(self.bottom)
        mask = self.grid.float()[None, None]
        high = F.max_pool2d(mask, 3, 1, 1)[0, 0]
        low = -F.max_pool2d(-mask, 3, 1, 1)[0, 0]
        self.inside = (high == low) & (grid >= 0)
        self.ids = torch.arange(n, device=dev)[:, None]
        # First other foreground strictly below the actual bottom.
        barrier = self.grid >= 0
        self.after_bottom = self.first_after(barrier, self.bottom)

    def first_after(self, mask, y):
        row = pixels(mask.device)[..., 1]
        first = torch.where(mask, row, H).flip((0,)).cummin(0).values.flip((0,))
        return F.pad(first, (0, 0, 0, 1), value=H)[(y.floor().long() + 1).clamp(0, H), self.x]

    def contact_bounds(self, road):
        first = self.first_after(road, self.hull)
        direct = (first < self.after_bottom) & (first < H) & (self.count > 0)
        # One straight edge has a common clearance bound. A visible road cell
        # limits that clearance at the last foreground row above the cell.
        segment = self.ids * W + self.facets
        gap = (first - 1. - self.hull).clamp_min(0).masked_fill(~direct, float("inf"))
        limit = gap.new_full((self.size,), float("inf"))
        limit.scatter_reduce_(0, segment.flatten(), gap.flatten(), reduce="amin")
        clearance = limit[segment]
        supported = (self.count > 0) & torch.isfinite(clearance)
        lower = torch.where(supported, self.hull, self.bottom)
        upper = torch.where(supported, self.hull + clearance, lower)
        return lower, upper, supported, direct.sum(1)


def plane_inverse(scene, plane, x, row):
    return (
        (x - scene.c[0]) / scene.f[0] * plane[0] + (row - scene.c[1]) / scene.f[1] * plane[1] + plane[2]
    )


def silhouette_membership(grid, owner, point):
    # Fraction of a bilinear footprint returning to the initiating center-frame silhouette.
    fixed = (point.float() * 32).round().long()
    ix, iy = fixed[:, 0] >> 5, fixed[:, 1] >> 5
    fx, fy = (fixed[:, 0] % 32).double() / 32, (fixed[:, 1] % 32).double() / 32

    def belongs(dy, dx):
        return grid[(iy + dy).clamp(0, H - 1), (ix + dx).clamp(0, W - 1)] == owner

    value = (belongs(0, 0) * (1-fx) * (1-fy) + belongs(0, 1) * fx * (1-fy)
             + belongs(1, 0) * (1-fx) * fy + belongs(1, 1) * fx * fy)
    return torch.where(visible(point), value, 0.)


def endpoint_agreement(scene, endpoint_flow, rows, cols, inverse_depth, cycle_variance, algebraic, grid, owners):
    """Test observed outer-frame motion against the ground-constrained stationary projection."""
    origin = pixels(scene.K.device)[rows, cols].double()
    ray = origin - scene.c
    short = origin + scene.flow[:, rows, cols]

    def rectify(view, point):
        rotated = homogeneous((point - scene.c) / scene.f) @ scene.poses[view, :, :3]
        good = visible(point) & (rotated[:, 2] > 0)
        return rotated[:, :2] * scene.f / torch.where(good, rotated[:, 2], 1.)[:, None], good

    observed = torch.stack([rectify(s, short[s])[0] for s in (0, 1)])
    static, front = [], []
    for pose in scene.poses:
        translation = pose[:, :3].T @ pose[:, 3]
        z = 1 + inverse_depth * translation[2]
        front.append(z > 0)
        projection = ray + inverse_depth[:, None] * translation[:2] * scene.f
        static.append(projection / torch.where(z > 0, z, 1.)[:, None])
    static = torch.stack(static)
    forward = front[0] & front[1]
    adjacent_noise = (static - ray).square().sum(-1) + cycle_variance + scene.road_variance[:, None]
    residual = (observed - static) / adjacent_noise.sqrt()[..., None]
    adjacent = torch.where(forward, -(residual[0] * residual[1]).sum(-1), algebraic)

    field = endpoint_flow.permute(0, 2, 3, 1).float()
    span = static[1] - static[0]
    power = span.square().sum(-1)
    membership = torch.ones_like(inverse_depth)
    errors, variances, visible_span = [], [], forward
    for start, sign in ((0, 1), (1, -1)):
        anchor = short[start]
        displacement = reverse_at(field[start], anchor)
        destination = anchor + displacement
        back = reverse_at(field[1-start], destination)
        returned = destination + reverse_at(scene.reverse_flow[1-start], destination)
        membership *= silhouette_membership(grid, owners, returned)
        point, seen = rectify(1-start, destination)
        errors.append(sign * (point - observed[start]) - span)
        variances.append((1./6 + (displacement + back).square().sum(-1) / 2).double())
        visible_span = visible_span & seen & torch.isfinite(displacement).all(-1) & torch.isfinite(back).all(-1)
    noise = power[None] + torch.stack(variances) + scene.road_variance.sum()
    agreement = (errors[0] * errors[1]).sum(-1) / noise.prod(0).sqrt()
    return torch.where(visible_span, membership * agreement + (1 - membership) * adjacent, adjacent)


def optical(obj, scene, endpoint_flow):
    yy, xx = torch.where(obj.inside & scene.visible)
    own = obj.grid[yy, xx]
    idx = own * W + xx
    aa = scene.par[:, yy, xx]
    obs = scene.obs[:, yy, xx]
    var = scene.var[:, yy, xx]
    power = aa.square().sum(-1)
    cross = (aa * obs).sum(-1)

    def fit(w):
        def col(v):
            return torch.zeros(obj.size, device=v.device, dtype=v.dtype).scatter_add_(0, idx, v.sum(0))

        den = col(w * power)
        return torch.where(den > 0, col(w * cross) / den, 0.0).reshape(obj.n, W)

    pilot = fit(1.0 / var)
    err = obs - aa * pilot[own, xx][None, :, None]
    u = fit(1.0 / (var + err.square().sum(-1)))
    low, upper, supported, columns = obj.contact_bounds(scene.road)
    planes = scene.planes

    def bounds(row):
        return torch.stack([plane_inverse(scene, p, obj.x, row) for p in planes])

    L = bounds(low - 0.5).nan_to_num(nan=float("inf")).amin(0).clamp_min(0)
    U = bounds(upper + 0.5).nan_to_num(nan=-float("inf")).amax(0)
    L = torch.where(torch.isfinite(L), L, 0.0)
    U = torch.where(supported & (U > 0), torch.maximum(L, U), float("inf"))
    depth = u.clamp(L, U)[own, xx]
    residual = obs - aa * depth[None, :, None]
    # Work in image space. The static parallax along a particular image ray
    # vanishes at the epipole even when the camera travels a long distance.
    parallax = aa * depth[None,:,None]
    noise = parallax.square().sum(-1) + var + scene.road_variance[:,None]
    product = -(residual[0]*residual[1]).sum(-1)/noise.prod(0).sqrt()
    product = endpoint_agreement(scene, endpoint_flow, yy, xx, depth, var, product, obj.grid, own)
    evidence = instance_median(product,own,obj.n).clamp_min(0).sqrt()
    pair = torch.stack([instance_median(e.norm(dim=-1)/v.sqrt(),own,obj.n)
                        for e,v in zip(residual,noise)])
    return pair,evidence,columns


def coverage(obj):
    # Chunk only along instances to bound temporary image storage.
    out = []
    for ids in obj.ids.flatten().split(16):
        mask = (obj.grid[None] == ids[:, None, None]).double()
        cells = F.avg_pool2d(mask, 2, 2)
        local = F.avg_pool2d(cells, 3, 1, 1)
        out.append((cells * local).sum((1, 2)) / cells.sum((1, 2)).clamp_min(0.25))
    return torch.cat(out)


def empty_evidence(n, device):
    z=torch.zeros(n,device=device,dtype=torch.float64)
    return dict(score=z.clone(), scores=z.new_zeros((2,n)), coverage=z.clone(),
                selected=z.bool(),contact=z.bool(),contact_columns=z.int(),
                road_h=z.new_zeros((2,3,3)),road_variance=z.new_zeros(2))


@torch.inference_mode()
def measure(road, K, flow, grid, groups, endpoint_flow):
    if not flow.is_cuda or any(v.device != flow.device for v in (road,K,grid,groups,endpoint_flow)):
        raise ValueError("GCDID needs all its inputs on one CUDA device")
    if flow.shape != (4,2,H,W) or endpoint_flow.shape != (2,2,H,W) or grid.shape != (H,W):
        raise ValueError("Unexpected GCDID input shape")
    out = empty_evidence(len(groups), flow.device)
    if not len(groups) or not (grid>=0).any():
        return out
    road = road & (grid < 0)
    fit = visual_scene(pixels(K.device),road,grid,K,flow,reverse_at,visible)
    if fit is None:
        return out
    poses, normal, variance = fit
    scene = Scene(road,K,poses,flow,normal,variance)
    obj = Instances(grid,len(groups))
    pairs, discrepancy, columns = optical(obj,scene,endpoint_flow)
    cov = coverage(obj)
    amplitude = cov * discrepancy
    out.update(score=amplitude/(1+amplitude), scores=torch.nan_to_num(pairs), coverage=cov,
               contact_columns=columns.int(), contact=columns>0,
               road_h=poses[:,:,:3]+poses[:,:,3:]*normal, road_variance=variance)
    return out
