# save_bev.py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------- utilities ----------
def _to_numpy_2d(x: torch.Tensor) -> np.ndarray:
    """(H,W) float32 numpy on CPU."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    assert x.ndim == 2, f"expected (H,W), got {x.shape}"
    return x

def _to_uint8(img: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    img = img.astype(np.float32)
    if vmin is None: vmin = np.nanmin(img)
    if vmax is None: vmax = np.nanmax(img)
    if vmax <= vmin:  # avoid div-by-zero
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - vmin) / (vmax - vmin)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)

def _apply_cmap(gray_u8: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """Return HxWx3 uint8 (RGB) from HxW uint8 using a matplotlib colormap."""
    cm = plt.get_cmap(cmap)
    rgb = cm(gray_u8.astype(np.float32) / 255.0)[..., :3]  # drop alpha
    return (rgb * 255.0 + 0.5).astype(np.uint8)

# ---------- main helpers ----------
def save_bev_single(
    bev: torch.Tensor,
    out_png: str,
    b: int = 0,
    mode: str = "mean",      # "channel", "mean", "max", "pca3"
    c: int = 0,
    cmap: str = "viridis",   # used for grayscale-to-RGB; ignored for pca3 (already RGB)
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    bev: torch.Tensor [B, C, 360, 360]
    mode:
      - "channel": save a specific channel c as heatmap
      - "mean"   : mean over channels
      - "max"    : max over channels
      - "pca3"   : RGB via first 3 PCA components over channels
    """
    assert bev.ndim == 4 and bev.shape[2:] == (360, 360), f"expected [B,C,360,360], got {bev.shape}"
    bev_b = bev[b]  # [C, H, W]

    if mode == "channel":
        mat = _to_numpy_2d(bev_b[c])
        gray_u8 = _to_uint8(mat, vmin=vmin, vmax=vmax)
        rgb = _apply_cmap(gray_u8, cmap=cmap)

    elif mode in ("mean", "max"):
        if mode == "mean":
            mat = _to_numpy_2d(bev_b.mean(dim=0))
        else:
            mat = _to_numpy_2d(bev_b.max(dim=0).values)
        gray_u8 = _to_uint8(mat, vmin=vmin, vmax=vmax)
        rgb = _apply_cmap(gray_u8, cmap=cmap)

    elif mode == "pca3":
        # bev_b: [C, H, W] -> reshape to [C, H*W], run PCA on channels
        C, H, W = bev_b.shape
        X = bev_b.detach().float().cpu().reshape(C, H * W).numpy()  # [C, N]
        X = X - X.mean(axis=1, keepdims=True)
        # covariance across channels (C x C), eigendecomp
        cov = (X @ X.T) / (X.shape[1] - 1 + 1e-6)
        vals, vecs = np.linalg.eigh(cov)         # ascending
        comp = vecs[:, -3:]                      # top-3
        Y = comp.T @ X                           # [3, N]
        Y = Y.reshape(3, H, W)
        # normalize each channel to 0..255 independently
        rgb = np.stack([_to_uint8(Y[i]) for i in range(3)], axis=-1)

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    Image.fromarray(rgb).save(out_png)
    print(f"Saved {out_png} ({mode})")