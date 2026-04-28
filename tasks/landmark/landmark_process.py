"""
tasks/landmark/landmark_process.py
高斯热图生成，原样迁移自 Cardiac_Landmark/landmark_process.py
"""
import numpy as np
from typing import Tuple, Union


def generate_gaussian_heatmaps(
    heatmap_size,
    keypoints: np.ndarray,
    sigma: Union[float, Tuple[float], np.ndarray],
) -> np.ndarray:
    """
    Generate Gaussian heatmaps for keypoints.
    Args:
        heatmap_size: (W, H)
        keypoints: (K, 2) array of (x, y) coordinates
        sigma: Gaussian sigma
    Returns:
        heatmaps: (K, H, W)
    """
    K, _ = keypoints.shape
    W, H = heatmap_size
    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    radius = sigma * 3

    gaussian_size = 2 * radius + 1
    x = np.arange(0, gaussian_size, 1, dtype=np.float32)
    y = x[:, None]
    x0 = y0 = gaussian_size // 2
    gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    for k in range(K):
        mu = (keypoints[k] + 0.5).astype(np.int64)
        left, top   = (mu - radius).astype(np.int64)
        right, bottom = (mu + radius + 1).astype(np.int64)

        g_x1 = max(0, -left);  g_x2 = min(W, right) - left
        g_y1 = max(0, -top);   g_y2 = min(H, bottom) - top
        h_x1 = max(0, left);   h_x2 = min(W, right)
        h_y1 = max(0, top);    h_y2 = min(H, bottom)

        np.maximum(heatmaps[k, h_y1:h_y2, h_x1:h_x2],
                   gaussian[g_y1:g_y2, g_x1:g_x2],
                   out=heatmaps[k, h_y1:h_y2, h_x1:h_x2])
    return heatmaps
