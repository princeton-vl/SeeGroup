import torch

def align_depth_least_square(
    gt_arr: torch.Tensor,
    pred_arr: torch.Tensor,
    valid_mask_arr: torch.Tensor,
    return_scale_shift=True,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # least square solver
    _ones = torch.ones_like(pred_masked)
    A = torch.cat([pred_masked, _ones], axis=-1)
    X = torch.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    return aligned_pred

def do_alignment(gt, pd, alignment, mask):
    batch_size = gt.shape[0]
    new_gt = torch.zeros_like(gt)
    new_pd = torch.zeros_like(pd)

    for i in range(batch_size):
        new_gt[i], new_pd[i] = do_alignment_single(gt[i:i+1], pd[i:i+1], alignment, mask[i:i+1])
    return new_gt, new_pd

def do_alignment_single(gt, pd, alignment, mask):
    if alignment == 'least_square':
        pd = align_depth_least_square(
            gt_arr=gt,
            pred_arr=pd,
            valid_mask_arr=(gt > 0) & (gt < 1e4) & (pd > 0) & (pd < 1e4) & mask,
            return_scale_shift=False,
        )
    elif alignment == 'normalization':
        gt = do_normalization(gt, mask)
        pd = do_normalization(pd, None)
    elif alignment == 'none':
        pass
    else:
        raise ValueError(f'Unsupported alignment: {alignment}')
    return gt, pd

def do_normalization(arr, mask=None):
    if mask is None:
        mask = (torch.ones_like(arr) > 0).to(arr.device)
    if mask.sum() == 0:
        return arr
    mask = mask.detach()
    d = arr[mask]

    # t = arr[mask]'s median
    t = torch.median(d)

    # s = abs(d - t).mean()
    s = torch.abs(d - t).mean() + 1e-6

    return (arr - t) / s
