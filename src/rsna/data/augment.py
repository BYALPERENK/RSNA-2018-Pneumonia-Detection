"""Batch augmentation on the GPU for grayscale images stored as uint8 (N, H, W) tensors."""
import math

import torch
import torch.nn.functional as F


def to_float(images: torch.Tensor) -> torch.Tensor:
    """uint8 (N, H, W) -> float (N, 1, H, W) in [0, 1]."""
    return images.unsqueeze(1).float().div_(255)


def augment(images: torch.Tensor, generator: torch.Generator, max_rotate: float = 10.0, scale=(0.9, 1.1),
            max_shift: float = 0.05, brightness: float = 0.1, contrast=(0.9, 1.1), vflip: bool = False,
            return_theta: bool = False):
    """Random horizontal flip, rotation, scale, shift, brightness and contrast; uint8 (N, H, W) -> float (N, 1, H, W).

    vflip adds a random vertical flip. return_theta also returns the (N, 2, 3) affine matrices, for transform_boxes.

    max_shift is a fraction of the image size. All random numbers come from `generator`, so a fixed seed gives
    the same augmentation whatever else uses the global random state.
    """
    x = to_float(images)
    n, device = len(x), x.device

    def uniform(low, high):
        return torch.rand(n, device=device, generator=generator) * (high - low) + low

    angle = uniform(-max_rotate, max_rotate) * math.pi / 180
    zoom = uniform(*scale)
    flip = torch.where(torch.rand(n, device=device, generator=generator) < 0.5, -1.0, 1.0)
    flip_v = torch.where(torch.rand(n, device=device, generator=generator) < 0.5, -1.0, 1.0) if vflip else torch.ones(n, device=device)
    shift = torch.stack([uniform(-max_shift, max_shift), uniform(-max_shift, max_shift)], dim=1) * 2  # grid spans [-1, 1]

    # affine_grid maps output pixels to input pixels, so zoom > 1 enlarges the image by sampling a smaller area
    cos, sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    theta = torch.stack([torch.stack([cos * flip, -sin * flip_v, shift[:, 0]], dim=1),
                         torch.stack([sin * flip, cos * flip_v, shift[:, 1]], dim=1)], dim=1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * uniform(*contrast).view(n, 1, 1, 1) + mean + uniform(-brightness, brightness).view(n, 1, 1, 1)
    x = x.clamp_(0, 1)
    return (x, theta) if return_theta else x


def transform_boxes(boxes: torch.Tensor, theta: torch.Tensor, size: int) -> torch.Tensor:
    """Move (n, 4) x1, y1, x2, y2 pixel boxes of one image with its augmentation matrix theta (2, 3) from augment().

    theta maps output to input coordinates, so its inverse is applied to the 4 corners. The result is the enclosing
    box (exact without rotation), clipped to the image.
    """
    a, t = theta[:, :2], theta[:, 2]
    x1, y1, x2, y2 = (boxes * (2 / size) - 1).unbind(1)  # pixels -> [-1, 1]
    corners = torch.stack([torch.stack(c, dim=1) for c in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))], dim=1)  # (n, 4, 2)
    out = (corners - t) @ torch.linalg.inv(a).T
    out = (out + 1) * (size / 2)
    return torch.cat([out.amin(1), out.amax(1)], dim=1).clamp(0, size)
