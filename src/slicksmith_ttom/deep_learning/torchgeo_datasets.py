import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import (
    IntersectionDataset,
    RasterDataset,
    concat_samples,
    random_bbox_assignment,
    stack_samples,
)
from torchgeo.samplers import GridGeoSampler

from .BalancedRandomGeoSampler import (
    BalancedRandomGeoSampler,
    build_integral_mask_from_raster_dataset,
)


class TtomLabelDataset(RasterDataset):
    """RasterDataset for the labels of public oil spill dataset found at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13761290.svg)](https://doi.org/10.5281/zenodo.13761290).

    Args:
        paths: one or more root directories to search or files to load
        crs: :term:`coordinate reference system (CRS)` to warp to
            (defaults to the CRS of the first file found)
        res: resolution of the dataset in units of CRS
            (defaults to the resolution of the first file found)
        bands: bands to return (defaults to all bands)
        transforms: a function/transform that takes an input sample
            and returns a transformed version
        cache: if True, cache file handle to speed up repeated sampling

    Raises:
        AssertionError: If *bands* are invalid.
        DatasetNotFoundError: If dataset is not found.

    """

    filename_glob = "*.tif"
    filename_regex = r"(?P<index>\d+)_(?P<date>\d{8}T\d{6}).tif"
    date_format = "%Y%m%dT%H%M%S"
    is_image = False


class TtomImageDataset(RasterDataset):
    """RasterDataset for the images of public oil spill dataset found at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13761290.svg)](https://doi.org/10.5281/zenodo.13761290).

    Args:
        paths: one or more root directories to search or files to load
        crs: :term:`coordinate reference system (CRS)` to warp to
            (defaults to the CRS of the first file found)
        res: resolution of the dataset in units of CRS
            (defaults to the resolution of the first file found)
        bands: bands to return (defaults to all bands)
        transforms: a function/transform that takes an input sample
            and returns a transformed version
        cache: if True, cache file handle to speed up repeated sampling

    Raises:
        AssertionError: If *bands* are invalid.
        DatasetNotFoundError: If dataset is not found.

    """

    filename_glob = "*.tif"
    filename_regex = r"(?P<index>\d+)_(?P<date>\d{8}T\d{6}).tif"
    date_format = "%Y%m%dT%H%M%S"
    is_image = True


class TtomDataModule(GeoDataModule):
    """
    Ttom remote-sensing segmentation datamodule.

    The module builds three GeoDataset splits (train/val/test), optionally
    samples a user-defined number of patches per split, and exposes
    PyTorch-Lightning‐ready DataLoaders.

    Args:
        train_img_path (os.PathLike): Directory or file containing training images.
        train_lbl_path (os.PathLike): Directory or file containing training labels.
        val_img_path (os.PathLike): Directory or file containing validation images.
        val_lbl_path (os.PathLike): Directory or file containing validation labels.
        test_img_path (os.PathLike): Directory or file containing test images.
        test_lbl_path (os.PathLike): Directory or file containing test labels.
        num_train_patches (int | None, optional): Patches drawn per training epoch.
            If an ``int`` the split uses ``BalancedRandomGeoSampler`` with
            ``pos_ratio``; if ``None`` the entire scene is enumerated via
            ``GridGeoSampler``. Defaults to ``1000``.
        num_val_patches (int | None, optional): Same logic as ``num_train_patches``
            for the validation split. Defaults to ``100``.
        num_test_patches (int | None, optional): Same logic as ``num_train_patches``
            for the test split. Defaults to ``1000``.
        num_examples (int, optional): Number of samples to log or visualise.
            Defaults to ``10``.
        batch_size (int, optional): Patches per mini-batch. Defaults to ``4``.
        patch_size (tuple[int, int], optional): Spatial size ``(h, w)`` of each
            patch in pixels. Defaults to ``(480, 480)``.
        split_only_train (bool, optional): If ``True``, only the training mosaic
            is subdivided and the other paths are ignored. Defaults to ``False``.
        num_workers (int | None, optional): Worker processes for each DataLoader.
            ``None`` falls back to ``os.cpu_count()``. Defaults to ``None``.
        pos_ratio (float, optional): Positive-class ratio supplied to
            ``BalancedRandomGeoSampler`` when patch counts are integers.
            Defaults to ``0.5``.
    """

    def __init__(
        self,
        train_img_path: os.PathLike,
        train_lbl_path: os.PathLike,
        val_img_path: os.PathLike,
        val_lbl_path: os.PathLike,
        test_img_path: os.PathLike,
        test_lbl_path: os.PathLike,
        num_train_patches: int | None = 1000,
        num_val_patches: int | None = 100,
        num_test_patches: int | None = 1000,
        num_examples=10,
        batch_size=4,
        patch_size=(480, 480),
        split_only_train=False,
        num_workers=None,
        pos_ratio: int = 0.5,
    ):
        super().__init__(
            dataset_class=IntersectionDataset,
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
        )
        self.train_img_path = train_img_path
        self.train_lbl_path = train_lbl_path
        self.val_img_path = val_img_path
        self.val_lbl_path = val_lbl_path
        self.test_img_path = test_img_path
        self.test_lbl_path = test_lbl_path
        self.num_train_patches = num_train_patches
        self.num_val_patches = num_val_patches
        self.num_test_patches = num_test_patches
        self.num_examples = num_examples
        self.split_only_train = split_only_train
        self.pos_ratio = pos_ratio

    def setup(self, stage="fit"):
        train_dataset = self._load_ttom_dataset(
            self.train_img_path, self.train_lbl_path
        )

        if self.split_only_train:
            ## for repeatability
            generator = torch.Generator().manual_seed(0)
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = random_bbox_assignment(train_dataset, [0.7, 0.1, 0.2], generator)
        else:
            self.train_dataset = train_dataset
            self.val_dataset = self._load_ttom_dataset(
                self.val_img_path, self.val_lbl_path
            )
            self.test_dataset = self._load_ttom_dataset(
                self.test_img_path, self.test_lbl_path
            )

        self._setup_samplers()

    def example_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            sampler=self.example_sampler,
            batch_size=self.num_examples,
            collate_fn=stack_samples,
        )

    def _setup_samplers(self):
        ## Train
        train_integral_mask, train_integral_transform = (
            build_integral_mask_from_raster_dataset(self.train_dataset.datasets[1])
        )
        self.train_sampler = BalancedRandomGeoSampler(
            self.train_dataset,
            self.patch_size,
            length=self.num_train_patches,
            pos_ratio=self.pos_ratio,
            integral_mask=train_integral_mask,
            integral_transform=train_integral_transform,
        )

        ## Val
        if self.num_val_patches is not None:
            val_integral_mask, val_integral_transform = (
                build_integral_mask_from_raster_dataset(self.val_dataset.datasets[1])
            )
            self.val_sampler = BalancedRandomGeoSampler(
                self.val_dataset,
                self.patch_size,
                length=self.num_val_patches,
                pos_ratio=self.pos_ratio,
                integral_mask=val_integral_mask,
                integral_transform=val_integral_transform,
            )
        else:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )

        ## Test
        if self.num_test_patches:
            test_integral_mask, test_integral_transform = (
                build_integral_mask_from_raster_dataset(self.test_dataset.datasets[1])
            )
            self.test_sampler = BalancedRandomGeoSampler(
                self.test_dataset,
                self.patch_size,
                length=self.num_test_patches,
                pos_ratio=self.pos_ratio,
                integral_mask=test_integral_mask,
                integral_transform=test_integral_transform,
            )
        else:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )

        ## Examples
        self.example_sampler = BalancedRandomGeoSampler(
            self.val_dataset,
            self.patch_size,
            length=self.num_examples,
            pos_ratio=self.pos_ratio,
        )

    @staticmethod
    def _load_ttom_dataset(img_path, lbl_path):
        return IntersectionDataset(
            dataset1=TtomImageDataset(
                img_path,
                transforms=image_transform,
            ),
            dataset2=TtomLabelDataset(
                lbl_path,
                transforms=label_transform,
            ),
            collate_fn=concat_samples,
        )


def image_transform(image):
    return image


def label_transform(label):
    return label


def save_examples(img_dir, lbl_dir, figures_dir: Path):
    # src_dir = Path("/Users/hjo109/Documents/data/Ttom")
    # img_dir = src_dir / "Oil_timestamped"
    # lbl_dir = src_dir / "Mask_oil_georef_timestamped"
    from slicksmith_ttom.vis import info_plots

    os.makedirs(figures_dir, exist_ok=True)
    img_ds = TtomImageDataset(img_dir)
    lbl_ds = TtomLabelDataset(lbl_dir)

    ds = IntersectionDataset(
        dataset1=img_ds,
        dataset2=lbl_ds,
        collate_fn=concat_samples,
    )

    samp = GridGeoSampler(ds, (512, 512), (512, 512))

    dl = DataLoader(
        ds,
        sampler=samp,
        batch_size=16,
        collate_fn=stack_samples,
    )

    for i, sample in enumerate(dl):
        img = sample["image"]
        mask = sample["mask"]
        print(f"{img.shape=}")
        print(f"{mask.shape=}")
        info_plots(img, mask, figures_dir / str(i))
        if i > 10:
            break


if __name__ == "__main__":
    img_path = Path("/Users/hjo109/Documents/data/Ttom/Oil_timestamped")
    lbl_path = Path("/Users/hjo109/Documents/data/Ttom/Mask_oil_georef_timestamped")

    save_examples(
        img_path,
        lbl_path,
        Path("/Users/hjo109/Documents/GitHub/slicksmith-ttom/output"),
    )
