"""Batch augmentation on the GPU for grayscale images stored as uint8 (N, H, W) tensors."""
import math

import torch
import torch.nn.functional as F


def to_float(images: torch.Tensor) -> torch.Tensor:
    """uint8 (N, H, W) -> float (N, 1, H, W) in [0, 1]."""
    return images.unsqueeze(1).float().div_(255)


def augment(images: torch.Tensor, generator: torch.Generator, max_rotate: float = 10.0, scale=(0.9, 1.1),
            max_shift: float = 0.05, brightness: float = 0.1, contrast=(0.9, 1.1), vflip: bool = False) -> torch.Tensor:
    """Random horizontal flip, rotation, scale, shift, brightness and contrast; uint8 (N, H, W) -> float (N, 1, H, W).

    vflip adds a random vertical flip.

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
    return x.clamp_(0, 1)
