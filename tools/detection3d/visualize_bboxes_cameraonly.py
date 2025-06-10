import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner, load_checkpoint
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the results of camera-only 3D detection models as a video in a 2x3 grid of images. "
        "The script projects 3D bounding boxes onto camera images and creates a visualization video."
    )
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold for predictions")
    parser.add_argument("--step", type=int, default=120, help="Number of steps to visualize")
    parser.add_argument("--cam_order", type=list, default=[2, 0, 4, 3, 1, 5], help="Camera order")
    return parser.parse_args()


def denormalize_image(img, mean, std, to_rgb=False):
    """
    Denormalize an image normalized with mean and std.
    Args:
        img (np.ndarray): Image array with shape (C, H, W) and dtype float32.
        mean (list): Per-channel mean.
        std (list): Per-channel std.
        to_rgb (bool): Whether to convert BGR to RGB after denormalization.
    Returns:
        np.ndarray: Denormalized image with shape (H, W, C) and dtype uint8.
    """
    assert img.shape[0] == 3, "Expected 3 channels (C, H, W)"
    denorm = img.copy()
    for i in range(3):
        denorm[i] = denorm[i] * std[i] + mean[i]
    denorm = denorm.transpose(1, 2, 0)  # (H, W, C)
    if to_rgb:
        denorm = denorm[..., ::-1]  # BGR to RGB
    return np.clip(denorm, 0, 255).astype(np.uint8)


def project_to_image(points, lidar2cam, cam2img):
    """Project 3D points to image plane."""
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = np.dot(lidar2cam, points_hom.T).T
    valid_mask = points_cam[:, 2] > 0
    points_img = np.dot(cam2img, points_cam[:, :3].T).T
    points_img /= points_img[:, 2:3]
    return points_img[:, :2], valid_mask


def draw_projected_3d_bboxes(ax, bboxes, lidar2cam, cam2img, img_shape):
    """
    Projects 3D bounding boxes to 2D and draws them on the given axis.
    Clips the lines at image boundaries.
    """
    H, W = img_shape[:2]

    def clip_line(pt1, pt2, W, H):
        """Clip a line segment to the image bounds."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Both points are inside
        if (0 <= x1 < W and 0 <= y1 < H) and (0 <= x2 < W and 0 <= y2 < H):
            return [pt1, pt2]

        # If both are out of bounds, skip
        if (x1 < 0 and x2 < 0) or (x1 >= W and x2 >= W) or (y1 < 0 and y2 < 0) or (y1 >= H and y2 >= H):
            return None

        # You can implement Liang-Barsky or use simple clamping for one-side clip
        pt1_clipped = [np.clip(x1, 0, W - 1), np.clip(y1, 0, H - 1)]
        pt2_clipped = [np.clip(x2, 0, W - 1), np.clip(y2, 0, H - 1)]
        return [pt1_clipped, pt2_clipped]

    for bbox in bboxes:
        x, y, z, dx, dy, dz, yaw = bbox.cpu().numpy()[:7]

        # Create 3D corners
        corners = np.array(
            [
                [dx / 2, dy / 2, -dz / 2],
                [dx / 2, -dy / 2, -dz / 2],
                [-dx / 2, -dy / 2, -dz / 2],
                [-dx / 2, dy / 2, -dz / 2],
                [dx / 2, dy / 2, dz / 2],
                [dx / 2, -dy / 2, dz / 2],
                [-dx / 2, -dy / 2, dz / 2],
                [-dx / 2, dy / 2, dz / 2],
            ]
        )

        # Apply rotation and translation
        R = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        rotated = corners @ R.T
        translated = rotated + np.array([x, y, z])

        # Project to image
        projected, valid_mask = project_to_image(translated, lidar2cam, cam2img)
        if not valid_mask.all():
            continue

        # Define lines
        lines = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

        for i, j in lines:
            pt1, pt2 = projected[i], projected[j]
            clipped = clip_line(pt1, pt2, W, H)
            if clipped:
                ax.plot([clipped[0][0], clipped[1][0]], [clipped[0][1], clipped[1][1]], color="lime", linewidth=1)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.work_dir = cfg.get("work_dir", "./work_dirs/default")

    runner = RUNNERS.build(cfg)
    load_checkpoint(runner.model, args.checkpoint)
    runner.model.eval()

    dloader = iter(runner.test_dataloader)
    os.makedirs(os.path.join(cfg.work_dir, "visualization"), exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(cfg.work_dir, "visualization", "visualization.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, 1.0, (1500, 1000))

    steps = args.step if args.step != -1 else len(dloader)
    for step in tqdm(range(steps)):  # change to desired number of samples
        data = next(dloader)
        with torch.no_grad():
            results = runner.model.test_step(data)

        result = results[0]
        pred_mask = result["pred_instances_3d"]["scores_3d"] > args.threshold
        pred_bboxes = result["pred_instances_3d"]["bboxes_3d"].tensor[pred_mask]

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        for i, cam_id in enumerate(args.cam_order):
            img = data["img"][0][0][cam_id].cpu().numpy()
            img = denormalize_image(img, cfg.img_norm_cfg.mean, cfg.img_norm_cfg.std, True).astype(np.uint8)

            lidar2cam = data["extrinsics"][0][0][cam_id].cpu().numpy()
            cam2img = data["intrinsics"][0][0][cam_id].cpu().numpy()

            ax = axs[i]
            ax.imshow(img)
            draw_projected_3d_bboxes(ax, pred_bboxes, lidar2cam, cam2img, img.shape)
            ax.axis("off")
            ax.set_title(f"Camera {cam_id}")

        plt.tight_layout()

        # Convert matplotlib figure to image
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Write frame to video
        video_writer.write(frame)
        plt.close()

    # Release video writer
    video_writer.release()
    print(f"Saved video to: {video_path}")


if __name__ == "__main__":
    main()
