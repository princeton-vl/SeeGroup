import torch
import torch.nn as nn
from util.align import do_alignment

from mmengine.registry import MODELS

class GradMatchingScaleLoss(nn.Module):
    def __init__(self, scale_level=4, alignment='none', align_all_layers=False):
        super(GradMatchingScaleLoss, self).__init__()
        self.scale_level = scale_level
        self.alignment = alignment
        self.align_all_layers = align_all_layers

    def forward(self, pred, targ, targ_mask):
        if self.alignment != 'none' and not self.align_all_layers:
            targ, pred = do_alignment(targ, pred, self.alignment, targ_mask)
        targ_mask = targ_mask.detach()

        diff = pred - targ
        diff = torch.mul(targ_mask, diff)

        loss = 0.0
        for scale in range(self.scale_level):
            window_size = 2 ** scale

            grad_x = torch.abs(diff[:, :, window_size:] - diff[:, :, :-window_size])
            mask_x = torch.mul(targ_mask[:, :, window_size:], targ_mask[:, :, :-window_size])
            grad_x = torch.mul(grad_x, mask_x)

            grad_y = torch.abs(diff[:, window_size:, :] - diff[:, :-window_size, :])
            mask_y = torch.mul(targ_mask[:, window_size:, :], targ_mask[:, :-window_size, :])
            grad_y = torch.mul(grad_y, mask_y)

            grad_x = torch.clamp(grad_x, max=100)
            grad_y = torch.clamp(grad_y, max=100)

            image_loss = torch.sum(grad_x) + torch.sum(grad_y)
            num_pixels = torch.sum(mask_x) + torch.sum(mask_y)

            if num_pixels > 0:
                loss += image_loss / num_pixels
            else:
                loss += 0 * image_loss
        return loss

def canonicalize_per_pixel(x, mask=None):
    x = torch.where(x < 1e-4, torch.ones_like(x) * 1e5, x)
    x_sorted, idx = torch.sort(x, dim=1)
    if mask is None:
        return x_sorted, None

    mask_sorted = torch.gather(mask, dim=1, index=idx)
    return x_sorted, mask_sorted


@MODELS.register_module()
class LayeredGradMatchingScaleLoss(nn.Module):
    def __init__(self, 
        scale_level=4, 
        alignment='none', 
        layer_weights=(1.0, 1.0, 1.0, 1.0),
        align_all_layers=False, 
        sort_pred=False, 
        sort_targ=False,
    ):
        super(LayeredGradMatchingScaleLoss, self).__init__()
        self.grad_matching_scale_loss = GradMatchingScaleLoss(scale_level, alignment, align_all_layers)
        self.layer_weights = layer_weights
        self.align_all_layers = align_all_layers
        self.alignment = alignment
        self.sort_pred = sort_pred
        self.sort_targ = sort_targ

    def forward(self, pred, targ, pred_mask=None, targ_mask=None):
        if pred_mask is None:
            pred_mask = torch.ones_like(pred).to(torch.bool)
        if targ_mask is None:
            targ_mask = torch.ones_like(targ).to(torch.bool)
        pred_mask = (pred_mask > 0.5).to(torch.bool)
        targ_mask = (targ_mask > 0.5).to(torch.bool)

        if self.sort_pred:
            pred = pred * pred_mask
            pred, pred_mask = canonicalize_per_pixel(pred, pred_mask)
        if self.sort_targ:
            targ = targ * targ_mask
            targ, targ_mask = canonicalize_per_pixel(targ, targ_mask)
        
        if self.alignment != 'none' and self.align_all_layers:
            targ, pred = do_alignment(targ, pred, self.alignment, targ_mask & pred_mask)
        
        loss = 0.0
        for i in range(pred.shape[1]):
            layered_loss = self.grad_matching_scale_loss(pred[:, i], targ[:, i], targ_mask[:, i])
            loss += layered_loss * self.layer_weights[i]
        return loss
