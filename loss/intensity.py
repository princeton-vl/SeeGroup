import math

import torch
from torch import nn

from util.align import do_alignment

from mmengine.registry import MODELS


@MODELS.register_module()
class IntensityLoss(nn.Module):
    def __init__(self, alignment='none', gamma=0.8, align_all_layers=False):
        super().__init__()
        self.alignment = alignment
        self.gamma = gamma
        self.align_all_layers = align_all_layers

    def likelihood_laplace(self, d, b, t):
        return 1.0 / (2.0 * b) * torch.exp(-torch.abs(d - t) / b)

    def nll_intensity(self, ds, bs, target, target_mask):
        targ_layer_num = target.shape[1]
        pred_layer_num = ds.shape[1]
        batch_size = target.shape[0]
        height, width = target.shape[2:]

        likelihoods = torch.zeros(
            batch_size,
            targ_layer_num,
            pred_layer_num,
            height,
            width,
            device=ds.device,
            dtype=ds.dtype,
        )

        for i in range(targ_layer_num):
            for j in range(pred_layer_num):
                likelihoods[:, i, j] = self.likelihood_laplace(ds[:, j], bs[:, j], target[:, i])
                likelihoods[:, i, j].masked_fill_(~target_mask[:, i], -math.inf)

        loss_targ_over_pred_total = 0.0
        loss_targ_over_pred_count = 0.0
        loss_pred_over_targ_total = 0.0
        loss_pred_over_targ_count = 0.0

        if self.gamma != 1.0:
            for i in range(targ_layer_num):
                max_likelihood, _ = torch.max(likelihoods[:, i], dim=1)
                nll = -torch.log(max_likelihood.clamp(min=1e-6))
                loss_targ_over_pred_total += (nll * target_mask[:, i]).sum()
                loss_targ_over_pred_count += target_mask[:, i].sum()
        loss_targ_over_pred = loss_targ_over_pred_total / (loss_targ_over_pred_count + 1e-6)

        if self.gamma != 0.0:
            for i in range(pred_layer_num):
                max_likelihood, _ = torch.max(likelihoods[:, :, i], dim=1)
                nll = -torch.log(max_likelihood.clamp(min=1e-6))
                loss_pred_over_targ_total += nll.sum()
                loss_pred_over_targ_count += target_mask.sum()
        loss_pred_over_targ = loss_pred_over_targ_total / (loss_pred_over_targ_count + 1e-6)

        return (
            (1.0 - self.gamma) * loss_targ_over_pred
            + self.gamma * loss_pred_over_targ
        )

    def forward(self, ds, bs, target, target_mask):
        if self.alignment != 'none' and self.align_all_layers:
            target, ds = do_alignment(target, ds, self.alignment, target_mask)

        return self.nll_intensity(ds, bs, target, target_mask)
