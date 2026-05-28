import numpy as np
import matplotlib.pyplot as plt

def _render_with_colorbar(arr, valid=None, vmin=None, vmax=None, cmap='Spectral'):
    # print(valid.sum(), valid.min(), valid.max())
    if valid is None:
        valid = np.isfinite(arr)
    
    if vmin is None:
        vmin = float(arr[valid].min()) if valid.any() else 0.0
    if vmax is None:
        vmax = float(arr[valid].max()) if valid.any() else 1.0
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

    # mask out invalid values
    arr = np.array(arr, copy=True)
    arr[~valid] = np.nan
    arr = np.ma.masked_invalid(arr)

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # already HxWx4
    plt.close(fig)
    return rgba[..., :3]

def visualize_depth(depth, valid=None, vmin=None, vmax=None):
    if valid is None:
        valid = (depth > 1e-3) & (depth < 1e4)

    return _render_with_colorbar(depth, valid, vmin, vmax)
