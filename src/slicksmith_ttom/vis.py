from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap


def _square_grid(n: int) -> int:
    """Return the side length of the smallest square grid that can fit *n* items."""
    return int(math.ceil(math.sqrt(n)))


def _ensure_rgb_or_single(img: np.ndarray) -> np.ndarray:
    """Convert a (C, H, W) NumPy tensor to RGB image in [0, 1] range."""
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.ndim > 3:
        raise ValueError("Expected (C, H, W) or (H, W) shape")

    c, h, w = img.shape

    if c == 1:
        img = np.repeat(img, 3, axis=0)  # grayscale → RGB
    elif c == 2:
        img = np.concatenate(
            [np.zeros((1, h, w)), img],
            axis=0,
        )
    elif c > 3:
        img = img[:3]  # take first three channels (PCA?)

    # scale to 0‒1 for display
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    return np.transpose(img, (1, 2, 0))  # → (H, W, 3)


def info_plots(x: torch.Tensor, y: torch.Tensor, path: os.PathLike) -> List[Path]:
    """Produce a battery of diagnostic plots for an image‐segmentation batch.

    The following assets are written to *path*:

    1. **image_grid_vv.png, image_grid_vh.png** – square-ish
        grid of *x* images with a shared colorbar.
    2. **mask_grid.png** – grid of segmentation masks with a
        discrete colorbar.
    3. **overlay_grid.png** – image / mask overlaps
        (RGB + 40 % alpha mask).
    4. **histograms_linear_vv.png, histograms_linear_vh.png**
        – per‑image intensity histograms (linear *y*).
    5. **histograms_log_vv.png, histograms_log_vh.png**
        – same histograms with logarithmic *y* scale.

    Parameters
    ----------
    x : torch.Tensor
        Image tensor shaped *(B, C, H, W)*.
    y : torch.Tensor
        Corresponding segmentation masks shaped
            *(B, H, W)* or *(B, 1, H, W)*.
    path : os.PathLike
        Output directory (will be created).
    """

    # ------------------------------------------------------------------
    # Handle inputs & filesystem
    # ------------------------------------------------------------------
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    x_np: np.ndarray = x.detach().cpu().numpy(force=True)
    y_np: np.ndarray = y.detach().cpu().numpy(force=True)

    if y_np.ndim == 4 and y_np.shape[1] == 1:
        y_np = y_np[:, 0]  # squeeze channel dim
    if y_np.ndim != 3:
        raise ValueError("y must be (B, H, W) or (B, 1, H, W)")

    b, c, h, w = x_np.shape
    side = _square_grid(b)

    saved: List[Path] = []

    vh = x_np[:, 0]
    vv = x_np[:, 1]
    # ------------------------------------------------------------------
    # 1. Image grid 1 -----------------------------------------------------
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(side, side, figsize=(side * 4, side * 3))
    axes = axes.ravel()
    last_im = None
    for idx, ax in enumerate(axes):
        if idx < b:
            img = _ensure_rgb_or_single(vv[idx])
            last_im = ax.imshow(
                img,
                vmin=vv.min(),
                vmax=vv.max(),
                cmap="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    fig.tight_layout()
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist())
    out = path / "image_grid_vv.png"
    fig.savefig(out, dpi=200)
    saved.append(out)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 1. Image grid 2 -----------------------------------------------------
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(side, side, figsize=(side * 4, side * 3))
    axes = axes.ravel()
    last_im = None
    for idx, ax in enumerate(axes):
        if idx < b:
            img = _ensure_rgb_or_single(vh[idx])
            last_im = ax.imshow(
                img,
                vmin=vv.min(),
                vmax=vv.max(),
                cmap="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    fig.tight_layout()
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist())
    out = path / "image_grid_vh.png"
    fig.savefig(out, dpi=200)
    saved.append(out)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Mask grid ------------------------------------------------------
    # ------------------------------------------------------------------
    n_classes: int = int(y_np.max()) + 1
    cmap = ListedColormap(plt.cm.get_cmap("tab10", n_classes).colors)

    fig, axes = plt.subplots(side, side, figsize=(side * 3, side * 3))
    axes = axes.ravel()
    last_im = None
    for idx, ax in enumerate(axes):
        if idx < b:
            last_im = ax.imshow(
                y_np[idx],
                cmap=cmap,
                vmin=0,
                vmax=n_classes - 1,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=axes.tolist(),
            fraction=0.015,
            pad=0.03,
            cmap=cmap,
        )
    fig.tight_layout()
    out = path / "mask_grid.png"
    fig.savefig(out, dpi=200)
    saved.append(out)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Overlay grid ---------------------------------------------------
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(side, side, figsize=(side * 4, side * 3))
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        if idx < b:
            ax.imshow(_ensure_rgb_or_single(x_np[idx]), cmap="gray")
            ax.imshow(
                y_np[idx],
                cmap=cmap,
                alpha=0.5,
                vmin=0,
                vmax=n_classes - 1,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    fig.tight_layout()
    out = path / "overlay_grid.png"
    fig.savefig(out, dpi=200)
    saved.append(out)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Histograms (linear & log) -------------------------------------
    # ------------------------------------------------------------------
    def _histograms(
        img_band_batch,
        band_name: str,
        log: bool = False,
    ) -> Path:
        fig, axes = plt.subplots(
            side,
            side,
            figsize=(side * 3, side * 3),
            sharex=True,
            sharey=True,
        )
        axes = axes.ravel()
        for idx, ax in enumerate(axes):
            if idx < b:
                data = img_band_batch[idx].flatten()
                ax.hist(data, bins=50)
                if log:
                    ax.set_yscale("log")
            else:
                ax.axis("off")
        fig.tight_layout()
        suffix = "log" if log else "linear"
        out = path / f"histograms_{suffix}_{band_name}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    saved.append(_histograms(vv, "vv", log=False))
    saved.append(_histograms(vv, "vv", log=True))
    saved.append(_histograms(vh, "vh", log=False))
    saved.append(_histograms(vh, "vh", log=True))

    return saved
