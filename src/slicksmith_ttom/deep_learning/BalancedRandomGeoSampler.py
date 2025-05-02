from __future__ import annotations

import contextlib
import random
from pathlib import Path
from typing import Iterator, Optional, Tuple
from warnings import warn
import rasterio
import torch
from affine import Affine
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers import RandomGeoSampler, get_random_bounding_box

__all__ = [
    "BalancedRandomGeoSampler",
    "build_integral_mask",
    "build_integral_mask_from_raster_dataset",
]


def build_integral_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert a binary 2‑D mask to its integral image with (1, 1) padding.

    Developed with ChatGPT.
    """
    mask = mask.to(torch.int32)
    integral = torch.cumsum(torch.cumsum(mask, dim=0), dim=1)
    return torch.nn.functional.pad(integral, (1, 0, 1, 0))


def _raster_files(ds):
    """Return a list of file paths for a TorchGeo RasterDataset.

    Developed with ChatGPT.
    """
    try:
        return ds.files  # type: ignore[attr-defined]
    except AttributeError as e:
        files = sorted(Path(ds.root).glob(ds.filename_glob))
        if not files:
            raise FileNotFoundError("No raster tiles found in dataset root") from e
        return files


# @timeit(runs=1, detailed=True)
def build_integral_mask_from_raster_dataset(
    label_ds,
    *,
    reference_ds: Optional[object] = None,
    band: int = 1,
    resampling: Resampling = Resampling.nearest,
    to_device: Optional[torch.device | str] = "cpu",
    res = 0.1, 
) -> Tuple[torch.Tensor, Affine]:
    """Mosaic **label rasters** into a single integral image.

    Parameters
    ----------
    label_ds : torchgeo.datasets.RasterDataset
        Dataset whose files contain 1‑band binary labels (oil slick = non‑zero).
    reference_ds : TorchGeo RasterDataset | rasterio dataset, optional
        If given, the label tiles are re‑projected to this dataset’s CRS so the
        resulting integral mask aligns with the imagery you will sample.
    band : int, default 1
        Band index (1‑based) containing the binary mask.
    resampling : rasterio.enums.Resampling, default *nearest*
        Resampling method used by ``WarpedVRT``.
    to_device : torch.device | str | None, default "cpu"
        Device for the returned tensor.
    res : tuple | int, default 0.1
        Output resolution in units of coordinate reference system. If not set, a source resolution will be used. If a single value is passed, output pixels will be square.

    Returns
    -------
    integral_mask : torch.Tensor
    integral_transform : Affine
        
    Developed with ChatGPT.
    """
    files = _raster_files(label_ds)
    if not files:
        raise ValueError("Label dataset contains no raster files")

    # –– Determine target CRS ––
    if reference_ds is not None:
        try:
            target_crs = reference_ds.crs  # TorchGeo datasets expose .crs
        except AttributeError:
            # Fallback: open first file of reference_ds and read its CRS
            ref_path = _raster_files(reference_ds)[0]
            with rasterio.open(str(ref_path)) as ref_src:
                target_crs = ref_src.crs
    else:
        with rasterio.open(str(files[0])) as src0:
            target_crs = src0.crs

    # –– Wrap each label tile, warping if needed ––
    srcs = []
    for fp in files:
        src = rasterio.open(str(fp))
        if src.crs != target_crs:
            src = WarpedVRT(src, crs=target_crs, resampling=resampling)
        srcs.append(src)

    mosaic, out_transform = rio_merge(srcs, res=0.1, nodata=0)
    # Cleanup (close datasets & VRTs)
    for s in srcs:
        with contextlib.suppress(Exception):
            s.close()
    mask_np = (mosaic[band - 1] != 0).astype("int32")

    mask = torch.from_numpy(mask_np)
    if to_device is not None:
        mask = mask.to(to_device)
    return build_integral_mask(mask), out_transform


class BalancedRandomGeoSampler(RandomGeoSampler):
    """RandomGeoSampler that yields a user‑specified
    ratio of positive to negative tiles.

    Parameters
    ----------
    dataset : torchgeo Dataset
        Any dataset that returns a dict with the key ``"mask"`` such that
        ``sample["mask"].sum() > 0`` means *oil present*.
    size : int | tuple[int, int]
        Tile size in pixels (see :class:`torchgeo.samplers.RandomGeoSampler`).
    pos_ratio : float, default 0.5
        Fraction of **positive** tiles in the long‑run output stream.
        Must be in ``(0, 1)``.  ``0.5`` → perfectly balanced.
    length : int | None, default None
        Effective number of samples the iterator will yield (keeps epochs
        consistent).  Pass ``None`` to fall back to
        ``RandomGeoSampler.default_length``.
    integral_mask : torch.Tensor | None, default None
        If provided, should be the output of :func:`build_integral_mask`.  A
        look‑up in this integral image replaces the expensive ``dataset[bbox]``
        read, speeding sampling by 1‑2 orders of magnitude.

    Developed with ChatGPT.
    """

    def __init__(
        self,
        dataset,
        size,
        pos_ratio: float = 0.5,
        integral_mask: Optional[torch.Tensor] = None,
        integral_transform: Optional[Affine] = None,
        length: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset=dataset,
            size=size,
            length=length,
            **kwargs,
        )

        if not 0.0 <= pos_ratio <= 1.0:
            raise ValueError("pos_ratio must lie strictly between 0 and 1")

        self.pos_ratio = float(pos_ratio)
        self.neg_ratio = 1.0 - self.pos_ratio
        self.integral_mask = integral_mask  # (H+1, W+1) or None
        self.integral_transform = integral_transform

        ## ^: xor
        if (integral_mask is not None) ^ (integral_transform is not None):
            raise ValueError(
                "integral_mask and integral_transform must be provided together"
            )

        if integral_transform is not None:
            # Precompute factors for CRS→pixel conversion (assume north‑up)
            a, _, c, _, e, f = integral_transform[:6]
            self._px_size_x = a
            self._px_size_y = -e  # e is negative for north‑up rasters
            self._origin_x = c
            self._origin_y = f

        ## Store dataset itself in the sampler to check
        ## for oil occurance if integral mask is not given.
        if integral_mask is None:
            self.dataset = dataset

    # ---------------------------------------------------------------------
    # Iterator interface
    # ---------------------------------------------------------------------
    def __iter__(self) -> Iterator[BoundingBox]:  # type: ignore[override]
        want_positive = random.random
        draw_bbox = self._draw_bbox
        N = len(self)
        i = 0
        while i < N:
            yield draw_bbox(want_positive() < self.pos_ratio)
            i += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _draw_bbox(self, want_pos: bool, max_iter=1000) -> BoundingBox:
        """Draw *one* bounding box that matches the requested class.

        Developed with ChatGPT.
        """
        is_pos = self._is_positive
        areas = self.areas
        hits = self.hits
        size = self.size
        res = self.res

        for _ in range(max_iter):
            print(",", end="")
            idx = torch.multinomial(areas, 1)
            hit = hits[idx]
            bbox = get_random_bounding_box(BoundingBox(*hit.bounds), size, res)
            if is_pos(bbox) == want_pos:
                return bbox  # ¯\\_(ツ)_/¯  finally
        warn(
            f"Tried {max_iter} random boxes without finding "
            "a positive patch so returned a negative one.",
            stacklevel=2,
        )
        return bbox

    # ---------------------------------------------------------------
    #  Class test (fast / slow path)
    # ---------------------------------------------------------------
    def _is_positive(self, bbox: BoundingBox) -> bool:
        if self.integral_mask is None:
            # Slow path: load sample; keep on CPU to avoid GPU thrash
            sample = self.dataset.datasets[1][bbox]
            return sample["mask"].sum().item() > 0

        # Fast O(1) via integral image
        # ---- CRS → pixel indices ----
        col0 = int((bbox.minx - self._origin_x) / self._px_size_x) + 1
        col1 = int((bbox.maxx - self._origin_x) / self._px_size_x) + 1
        row0 = int((self._origin_y - bbox.maxy) / self._px_size_y) + 1
        row1 = int((self._origin_y - bbox.miny) / self._px_size_y) + 1

        I = self.integral_mask  # noqa: E741
        # Clip to valid range to guard against numeric spill‑over
        col0 = max(1, min(col0, I.shape[1] - 1))
        col1 = max(1, min(col1, I.shape[1] - 1))
        row0 = max(1, min(row0, I.shape[0] - 1))
        row1 = max(1, min(row1, I.shape[0] - 1))

        s = (
            I[row1, col1]
            - I[row0 - 1, col1]
            - I[row1, col0 - 1]
            + I[row0 - 1, col0 - 1]
        )
        return s.item() > 0
