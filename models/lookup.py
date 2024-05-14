import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np




def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)
def look(coords,corr):

    r= 4
    coords = coords[:, :1].permute(0, 2, 3, 1)
    batch, h1, w1, _ = coords.shape
    dx = torch.linspace(-r, r, 2 * r + 1)
    dx = dx.view(1, 1, 2 * r + 1, 1).to(coords.device)
    x0 = dx + coords.reshape(batch * h1 * w1, 1, 1, 1)
    y0 = torch.zeros_like(x0)
    coords_lvl = torch.cat([x0, y0], dim=-1)
    corr = bilinear_sampler(corr, coords_lvl)
    corr = corr.view(batch, h1, w1, -1)

    return corr.permute(0,3,1,2).contiguous().float()

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """

    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img