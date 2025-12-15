# save_bev.py
import torch
import numpy as np
import pickle as pkl
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


def sample_augmentation(img, resize_lim, final_dim, bot_pct_lim, rot_lim):
    W, H = img.size
    print(f"H: {H}, W: {W}")
    fH, fW = final_dim

    if isinstance(resize_lim, (int, float)):
        aspect_ratio = max(fH / H, fW / W)
        resize = np.random.uniform(aspect_ratio, aspect_ratio + resize_lim)
    else:
        resize = np.random.uniform(*resize_lim)

    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims

    crop_h = int((1 - np.random.uniform(*bot_pct_lim)) * newH) - fH
    crop_w = int(np.random.uniform(0, max(0, newW - fW)))
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    # if self.rand_flip and np.random.choice([0, 1]):
    #     flip = True
    rotate = np.random.uniform(*rot_lim)
    return resize, resize_dims, crop, flip, rotate

def img_transform(img, rotation, translation, resize, resize_dims, crop, flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate, resample=Image.BICUBIC)  # Default rotation introduces artifacts.

    # post-homography transformation
    rotation *= resize
    translation -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([(crop[2] - crop[0]), 0])
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b
    theta = rotate / 180 * np.pi
    A = torch.Tensor(
        [
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ]
    )

    b = torch.Tensor([(crop[2] - crop[0]), (crop[3] - crop[1])]) / 2
    b = A.matmul(-b) + b
    rotation = A.matmul(rotation)
    translation = A.matmul(translation) + b

    return img, rotation, translation

if __name__ == "__main__":

    pickle_file = "data/t4dataset/info/kokseang_2_5/t4dataset_j6gen2_base_infos_train.pkl"

    with open(pickle_file, "rb") as fp:
        data = pkl.load(fp)
    
    resize_lim = [0.35, 0.35]
    final_dim = [384, 768]
    bot_pct_lim = [0.0, 0.0]
    rot_lim = [0.0, 0.0]
    camera_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]    
    for index, data_list in enumerate(data["data_list"]):
        imgs = []
        for camera_type in camera_order:
            if camera_type not in data_list["images"]:
                continue

            cam_item = data_list["images"][camera_type]
            img_path = "data/t4dataset/" + cam_item["img_path"]
            with Image.open(img_path, "r") as im:
                img = im.convert("RGB") 
                resize, resize_dims, crop, flip, rotate = sample_augmentation(im, resize_lim, final_dim, bot_pct_lim, rot_lim)
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                img, rotation, translation = img_transform(
                    img, 
                    post_rot, 
                    post_tran,
                    resize,
                    resize_dims,
                    crop,
                    flip, 
                    rotate
                )
                imgs.append(img)
        
        # Visualize image
        # -----------------------------
        # 🔵 Save as subplot with 5 images
        # -----------------------------
        num_images = len(imgs)
        cols = 5
        rows = int(np.ceil(num_images / cols))

        fig = plt.figure(figsize=(20, 4 * rows))

        for idx, img in enumerate(imgs):
            ax = fig.add_subplot(rows, cols, idx + 1)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Camera {idx}")

        fig.tight_layout()

        # # Save figure to memory (as ndarray) or file
        fig_path = f"work_dirs/bevfusion_image_2d_debug/4/{index}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
