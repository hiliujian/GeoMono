"""Estimate camera motion against the visible road and static background on CUDA."""
from functools import lru_cache
import json
from pathlib import Path
import torch

POSE_ITERATIONS = 4


def homogeneous(x):
    return torch.cat((x, torch.ones_like(x[..., :1])), -1)


@lru_cache(maxsize=1)
def road_prior():
    return json.loads(Path(__file__).with_name("road_dispersion.json").read_text())


def skew(x):
    a, b, c = x.unbind(-1)
    zero = torch.zeros_like(a)
    return torch.stack((zero, -c, b, c, zero, -a, -b, a, zero), -1).reshape(*a.shape, 3, 3)


def rotation(omega):
    angle = omega.norm()
    k = skew(omega)
    return (torch.eye(3, device=k.device, dtype=k.dtype) + torch.sinc(angle / torch.pi) * k
            + .5 * torch.sinc(angle / (2*torch.pi)).square() * (k @ k))


def visual_scene(xy, road, grid, K, flow, reverse_at, visible):
    """The road inverse depth in camera-height units is n.r; background depths are profiled out."""
    from .camera import NORMALIZED_K
    height, width = road.shape
    f, c = K.diag()[:2], K[:2, 2]
    nominal = K.new_tensor(NORMALIZED_K)[:3, :3]
    nominal[0] *= width
    nominal[1] *= height
    prior = road_prior()
    transform = K.T @ torch.linalg.inv(nominal.T)
    normal = transform @ K.new_tensor(prior["normal"])
    cov = transform @ K.new_tensor(prior["covariance"]) @ transform.T / normal.square().sum()
    normal = normal / normal.norm()
    sample = (grid < 0) & ((xy % 4) == 0).all(-1)
    rows, cols = torch.where(sample)
    src = xy[rows, cols].double()
    rays = homogeneous((src-c)/f)
    ground = road[rows, cols] & ((rays @ normal) > 0)
    # Half of the road lattice supplies out-of-fit reprojection uncertainty.
    fitting = ((rows//4 + cols//4) % 2 == 0)
    poses, observations = [], []
    eye = torch.eye(3, device=K.device, dtype=K.dtype)
    for s in range(2):
        fw = flow[2*s, :, rows, cols].T.double()
        dst = src + fw
        back = reverse_at(flow[2*s+1].permute(1,2,0).float(), dst)
        var = 1/6 + (fw+back).square().sum(1)/2
        ok = visible(dst) & torch.isfinite(var) & torch.isfinite(dst).all(1)
        train = ok & (fitting | ~ground)
        if int((ground & train).sum()) < 3:
            return None
        ray, q, is_road = rays[train], (dst[train]-c)/f, ground[train]
        variance = var[train]
        anchor = (ray @ normal).clamp_min(0)
        R, t = eye, K.new_zeros(3)
        def refine(indices, R, t, joint):
            p0, obs, g = ray[indices], q[indices], is_road[indices]
            rp = p0 @ R.T
            direction = (t[:2] - obs*t[2])*f
            right = (obs*rp[:,2:] - rp[:,:2])*f
            power = direction.square().sum(1)
            inverse = ((direction*right).sum(1)/power.clamp_min(torch.finfo(K.dtype).eps)).clamp_min(0)
            u = torch.where(g, anchor[indices], inverse)
            point = rp + u[:,None]*t
            z = point[:,2]
            front = z > torch.finfo(K.dtype).eps
            z = torch.where(front,z,1.)
            proj = point[:,:2]/z[:,None]
            D = torch.cat((eye[:2,:2].expand(len(p0),-1,-1), -proj[:,:,None]),2) * (f[None,:,None]/z[:,None,None])
            residual = (proj-obs)*f
            jac = torch.cat((D @ -skew(rp), u[:,None,None]*D),2)
            if joint:
                # An unanchored background point contributes only the component
                # orthogonal to the depth derivative, including during pure rotation.
                tangent = D @ t
                den = tangent.square().sum(1)
                free = (~g) & (inverse>0) & (den>torch.finfo(K.dtype).eps)
                v = tangent/torch.where(free,den,1.)[:,None].sqrt()
                v = torch.where(free[:,None],v,0.)
                residual = residual-v*(residual*v).sum(1,keepdim=True)
                jac = jac-v[:,:,None]*(v[:,:,None]*jac).sum(1,keepdim=True)
            weights = front / (variance[indices]+residual.square().sum(1))
            j = jac.reshape(-1,6)
            jw = (jac*weights[:,None,None]).reshape(-1,6).T
            delta = -torch.linalg.pinv(jw@j,hermitian=True)@(jw@residual.flatten())
            return rotation(delta[:3])@R, t+delta[3:]
        road_ids = torch.where(is_road)[0]
        all_ids = torch.arange(len(ray),device=K.device)
        for _ in range(POSE_ITERATIONS):
            R,t=refine(road_ids,R,t,False)
        for _ in range(POSE_ITERATIONS):
            R,t=refine(all_ids,R,t,True)
        if not torch.isfinite(R).all() or not torch.isfinite(t).all():
            return None
        observations.append((ok, dst, var))
        poses.append(torch.cat((R,t[:,None]),1))
    # Profile a common road inverse-depth plane for both time directions. The
    # covariance of road slopes was measured from unlabelled training images.
    precision = torch.linalg.pinv(cov,hermitian=True)
    terms = []
    for T,(ok,dst,var) in zip(poses,observations):
        use = ok & ground & fitting
        ray = rays[use]
        obs = (dst[use]-c)/f
        rp = ray @ T[:,:3].T
        t = T[:,3]
        vector = (t[:2]-obs*t[2])*f
        offset = (obs*rp[:,2:]-rp[:,:2])*f
        terms.append((ray,vector,offset,var[use]))
    plane = normal
    for _ in range(3):
        lhs, rhs = precision.clone(), precision @ normal
        for ray,vector,offset,variance in terms:
            residual = offset-vector*(ray@plane)[:,None]
            weight = 1/(variance+residual.square().sum(1))
            lhs += (ray.T*(weight*vector.square().sum(1)))@ray
            rhs += ray.T@(weight*(vector*offset).sum(1))
        plane = torch.linalg.solve(lhs,rhs)
    if not torch.isfinite(plane).all():
        return None
    valid = torch.stack([ok for ok, _, _ in observations])
    q = torch.stack([torch.where(ok[:, None], (dst - c) / f, 0.) for ok, dst, _ in observations])
    var = torch.stack([torch.where(ok, v, 1.) for ok, _, v in observations])
    poses = torch.stack(poses)
    refined, shared_plane = refine_background(rays, q, var, valid, ground, fitting, f,
                                               poses, plane, normal, precision)
    error = held_out_error(refined, shared_plane, rays, q, valid, ground, fitting, f)
    if error is not None:
        return refined, shared_plane, error
    error = held_out_error(poses, plane, rays, q, valid, ground, fitting, f)
    return (poses, plane, error) if error is not None else None


def held_out_error(poses, plane, rays, q, valid, ground, fitting, focal):
    variances = []
    for view, pose in enumerate(poses):
        road = valid[view] & ground
        hold = road & ~fitting
        if not hold.any():
            hold = road
        homography = pose[:, :3] + pose[:, 3:] * plane
        projected = rays[hold] @ homography.T
        front = projected[:, 2] > 0
        if not front.any():
            return None
        xy = projected[front, :2] / projected[front, 2:]
        residual = (xy - q[view, hold][front]) * focal
        variances.append(residual.square().sum(-1).median() / 2)
    return torch.stack(variances)


def refine_background(rays, q, variance, valid, ground, fitting, focal, poses, plane, normal, precision):
    """Fit two camera motions and a common plane after eliminating each background ray's depth."""
    mask = valid & (~ground | fitting)[None]
    eye = torch.eye(3, dtype=rays.dtype, device=rays.device)
    rotations, translations = poses[:, :, :3], poses[:, :, 3]
    initial_plane = plane
    eps = torch.finfo(rays.dtype).eps
    for _ in range(POSE_ITERATIONS):
        rotated = rays @ rotations.transpose(1, 2)
        direction = (translations[:, None, :2] - q * translations[:, None, 2:]) * focal
        offset = (q * rotated[:, :, 2:] - rotated[:, :, :2]) * focal

        def depth(weights):
            norm = (weights * direction.square().sum(-1)).sum(0)
            cross = (weights * (direction * offset).sum(-1)).sum(0)
            return (cross / norm.clamp_min(eps)).clamp_min(0.)

        free_depth = depth(mask / variance)
        error = offset - direction * free_depth[None, :, None]
        free_depth = depth(mask / (variance + error.square().sum(-1)))
        inverse = torch.where(ground, rays @ plane, free_depth)
        point = rotated + translations[:, None, :] * inverse[None, :, None]
        visible = mask & (point[:, :, 2] > eps)
        z = torch.where(visible, point[:, :, 2], 1.)
        xy = point[:, :, :2] / z[..., None]
        residual = (xy - q) * focal

        project = torch.cat((eye[:2, :2].expand(*xy.shape[:2], -1, -1), -xy[..., None]), -1)
        project *= focal[None, None, :, None] / z[:, :, None, None]
        depth_jac = (project @ translations[:, None, :, None]).squeeze(-1)
        local = torch.cat((project @ -skew(rotated), project * inverse[None, :, None, None]), -1)
        # The motion of each camera has six coordinates, shared road geometry has three.
        jac = rays.new_zeros((*xy.shape, 15))
        jac[0, :, :, :6], jac[1, :, :, 6:12] = local[0], local[1]
        jac[:, :, :, 12:] = depth_jac[..., None] * rays[None, :, None, :] * ground[None, :, None, None]
        weight = visible / (variance + residual.square().sum(-1))
        weighted = (jac * weight[:, :, None, None]).reshape(-1, 15)
        hessian = jac.reshape(-1, 15).T @ weighted
        gradient = weighted.T @ residual.flatten()

        strength = (weight * depth_jac.square().sum(-1)).sum(0)
        free = ~ground & (free_depth > 0) & (strength > eps)
        scale = torch.where(free, 1 / strength.clamp_min(eps), 0.)
        coupling = ((weight[:, :, None] * depth_jac)[..., None] * jac).sum((0, 2))
        depth_gradient = (weight * (depth_jac * residual).sum(-1)).sum(0)
        hessian -= coupling.T @ (coupling * scale[:, None])
        gradient -= coupling.T @ (depth_gradient * scale)
        hessian[12:, 12:] += precision
        gradient[12:] += precision @ (plane - normal)

        step = -torch.linalg.pinv(hessian, hermitian=True) @ gradient
        motion = step[:12].reshape(2, 6)
        rotations = torch.stack([rotation(delta[:3]) @ r for delta, r in zip(motion, rotations)])
        translations = translations + motion[:, 3:]
        plane = plane + step[12:]
    result = torch.cat((rotations, translations[:, :, None]), 2)
    if torch.isfinite(result).all() and torch.isfinite(plane).all():
        return result, plane
    return poses, initial_plane
