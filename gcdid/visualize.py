"""TensorBoard panels for motion selection and contact depth supervision."""
from functools import lru_cache

import torch


def overlay(rgb,mask,color):
    tint=rgb.new_tensor(color)[:,None,None]
    return torch.where(mask[None],.35*rgb+.65*tint,rgb)


def panel_order(inputs):
    """Choose the same batch items for every TensorBoard image slot."""
    count = min(4, len(inputs['color', 0, 0]))
    if 'dynamic_mask' not in inputs:
        return list(range(count))
    area = inputs['dynamic_mask'].flatten(1).sum(1)
    return area.argsort(descending=True)[:count].tolist()


@torch.no_grad()
def log_panels(writer,inputs,outputs,step,order=None):
    if 'gcdid/photo_excluded' not in outputs:
        return
    moving=inputs['dynamic_mask'].bool()
    excluded=outputs['gcdid/photo_excluded']
    removed_active=outputs['gcdid/photo_removed']
    raw_photo=outputs['gcdid/photo_raw']
    key=('raw_color',0,0) if ('raw_color',0,0) in inputs else ('color',0,0)
    order = panel_order(inputs) if order is None else order
    for slot,index in enumerate(order):
        rgb=inputs[key][index].detach().float().clamp(0,1)
        candidate=inputs['gcdid_grid'][index]>=0
        final_view=overlay(rgb,moving[index],[.1,.95,.2])
        excluded_view=overlay(rgb,excluded[index],[1.,0.,1.])
        raw=raw_photo[index].float()
        scale=torch.quantile(raw.flatten(),.95).clamp_min(1e-5)
        heat=torch.stack(((raw/scale).clamp(0,1),torch.zeros_like(raw),1-(raw/scale).clamp(0,1)))
        panel=torch.cat([rgb,overlay(rgb,candidate,[.6,.6,.6]),
                         final_view,excluded_view,heat],dim=2)
        writer.add_image('gcdid/panel_{}'.format(slot),panel,step)
        writer.add_text('gcdid/frame_{}'.format(slot),inputs['frame_key'][index],step)
        writer.add_image('gcdid/final_mask_{}'.format(slot),moving[index][None].float(),step)
        writer.add_image('gcdid/excluded_photo_{}'.format(slot),excluded[index][None].float(),step)
        writer.add_image('gcdid/removed_active_photo_{}'.format(slot),removed_active[index][None].float(),step)
        yy,xx=torch.where(moving[index])
        if yy.numel():
            y0=max(int(yy.min())-8,0);y1=min(int(yy.max())+9,rgb.shape[1])
            x0=max(int(xx.min())-8,0);x1=min(int(xx.max())+9,rgb.shape[2])
            zoom=torch.cat([rgb[:,y0:y1,x0:x1],final_view[:,y0:y1,x0:x1],
                            excluded_view[:,y0:y1,x0:x1]],dim=2)
            writer.add_image('gcdid/zoom_{}'.format(slot),zoom,step)
    writer.add_text('gcdid/panel_legend','RGB | candidate instances (grey) | moving instances (green) | full photometric exclusion (magenta) | raw photometric error',step)


@lru_cache(maxsize=4)
def magma_table(device):
    from matplotlib import pyplot

    colors = pyplot.get_cmap('magma', 256).colors
    rgb = torch.as_tensor(colors, device=device, dtype=torch.float64)[:, :3]
    return (rgb * 255).byte()


def colorize_depth(depth, low, high):
    normalized = ((high - depth) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    indices = (normalized * 256).long().clamp_max(255)
    return magma_table(depth.device)[indices].permute(2, 0, 1)


@lru_cache(maxsize=128)
def label_strip(width, height, text, device):
    import cv2
    import numpy as np

    # Rasterize the static caption once. Apply it to the image on the GPU.
    banner = np.empty((height, width, 3), dtype=np.uint8)
    banner[:] = (11, 18, 27)
    cv2.putText(banner, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.49,
                (245, 245, 245), 1, cv2.LINE_AA)
    return torch.from_numpy(banner).to(device).permute(2, 0, 1)


def titled_panel(images, labels):
    height, width = images[0].shape[-2:]
    caption_height = min(height, 24)
    panels = [torch.cat((label_strip(width, caption_height, label, image.device),
                         image[:, caption_height:]), dim=1)
              for image, label in zip(images, labels)]
    return torch.cat(panels, dim=2).float() / 255


@torch.no_grad()
def log_gcidc_panel(writer, inputs, outputs, index, slot, step):
    if 'dynamic_mask' not in inputs or 'gcidc/mask' not in outputs:
        return

    from gcidc import EPS

    depth = outputs['depth', 0, 0]
    depth = depth[:, 0] if depth.ndim == 4 else depth
    height, width = depth.shape[-2:]
    moving = inputs['dynamic_mask'][index].bool()
    if moving.shape != (height, width):
        return

    z = depth[index].detach().clamp_min(EPS)
    mask = outputs['gcidc/mask'][index]
    target = outputs['gcidc/u_t'][index]
    u = 1 / z
    pull = torch.where(mask, (target - u) / u.clamp_min(EPS), torch.zeros_like(z))
    target_depth = 1 / target.clamp_min(EPS)

    rgb = inputs.get(('raw_color', 0, 0), inputs['color', 0, 0])[index].float()
    rgb = (rgb.clamp(0, 1) * 255).byte()
    limits = torch.quantile(z.flatten().float(), z.new_tensor([0.05, 0.95], dtype=torch.float32))
    predicted = colorize_depth(z, *limits)
    completed = colorize_depth(torch.where(mask, target_depth, z), *limits)
    completed = completed.masked_fill(moving & ~mask, 65)
    green = z.new_tensor([26, 242, 51], dtype=torch.float64)[:, None, None]
    selection = torch.where(moving, 0.35 * rgb.double() + 0.65 * green, rgb).byte()

    magnitude = pull[mask].abs().float()
    spread = torch.quantile(magnitude, 0.9).clamp_min(0.05) if magnitude.numel() else z.new_tensor(0.05)
    relative = (pull / (1.6 * spread)).clamp(-1, 1)
    nearer = z.new_tensor([0.16, 0.38, 0.90], dtype=torch.float32)[:, None, None]
    farther = z.new_tensor([0.88, 0.22, 0.18], dtype=torch.float32)[:, None, None]
    color = torch.where(relative > 0, nearer, farther)
    alpha = (0.85 * relative.abs() * mask).float()
    blend = rgb.double() / 255 * (1 - alpha) + color * alpha
    direction = (blend.clamp(0, 1) * 255).byte()

    images = (rgb, selection, predicted, completed, direction)
    labels = ('Target RGB', 'GCDID selection | green', 'Pred depth | near bright',
              'GCIDC target | near bright', 'Depth pull | blue near, red far')
    writer.add_image(f'gcidc/panel_{slot}', titled_panel(images, labels), step)
    writer.add_text(f'gcidc/frame_{slot}', inputs['frame_key'][index], step)

    y, x = torch.where(moving | mask)
    if y.numel():
        pad = 6
        y0, y1 = max(int(y.min()) - pad, 0), min(int(y.max()) + pad, height)
        x0, x1 = max(int(x.min()) - pad, 0), min(int(x.max()) + pad, width)
        scale = min(5, max(1, round(110 / max(y1 - y0, 1))))
        crops = [image[:, y0:y1, x0:x1].repeat_interleave(scale, dim=1)
                 .repeat_interleave(scale, dim=2) for image in images]
        labels = [f'{label} x{scale}' for label in ('RGB', 'GCDID', 'Pred', 'GCIDC', 'Pull')]
        writer.add_image(f'gcidc/zoom_{slot}', titled_panel(crops, labels), step)

    added = mask & ~outputs['gcidc/original_mask'][index]
    writer.add_scalar(f'gcidc/coverage_{slot}', (mask & moving).sum().float() / moving.sum().clamp_min(1), step)
    writer.add_scalar(f'gcidc/target_px_{slot}', mask.sum(), step)
    writer.add_scalar(f'gcidc/added_target_px_{slot}', added.sum(), step)
    writer.add_scalar(f'gcidc/pull_{slot}', magnitude.mean() if magnitude.numel() else 0., step)
    writer.add_scalar(f'gcidc/pull_p90_{slot}', spread if magnitude.numel() else 0., step)
    writer.add_scalar(f'gcidc/target_depth_p50_{slot}', target_depth[mask].median() if magnitude.numel() else 0., step)
