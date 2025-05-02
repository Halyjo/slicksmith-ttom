import os
from datetime import datetime
from pathlib import Path

import py7zr
from tap import Tap
from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset, concat_samples, stack_samples
from torchgeo.samplers import GridGeoSampler

from slicksmith_ttom.deep_learning import (
    TtomImageDataset,
    TtomLabelDataset,
    BalancedRandomGeoSampler,
    build_integral_mask_from_raster_dataset,
)
from slicksmith_ttom.download import download_file
from slicksmith_ttom.preprocessing.add_georef_and_timestamps import (
    georeference_and_timestamp_images_and_masks,
)
from slicksmith_ttom.utils import save_console_outputs
from slicksmith_ttom.vis import info_plots

## Constants
DATA_SOURCE_URLS = dict(
    train_val_masks="https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_mask.7z",
    train_val_lookalike_mask="https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_mask.7z",
    train_val_images="https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_images.7z",
    train_val_lookalike_images="https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_images.7z",
    train_val_no_oil_images="https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_Images.7z",
    train_val_no_oil_mask="https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_mask.7z",
    test_images_masks="https://zenodo.org/records/13761290/files/02_Test_images_and_ground_truth.7z",
)


## Arguments Parser
class MyArgs(Tap):
    ## Paths
    download_dst: Path  # = Path("/storage/experiments/data/Ttom/")
    georef_and_timestamp_dst: Path  # = Path(
    #     "/storage/experiments/data/Ttom_torchgeo/"
    # )
    figures_dir: Path  # = Path("/storage/experiments/data/Ttom_examples/")

    ## Tasks
    download: bool = True
    process_for_torchgeo: bool = True
    make_info_plots: bool = True

    def process_args(self):
        self.examples_img_src = self.georef_and_timestamp_dst / "Oil_timestamped"
        self.examples_lbl_src = (
            self.georef_and_timestamp_dst / "Mask_oil_georef_timestamped"
        )


def main():
    """Does 3 things:

    Main
    ----
    1. download data
    2. Process for torchgeo
        - georeference labels and
        - add artificial timestamps to both imgs and lbls
    3. make info plots to better understand the data

    Use this with typed-argument-parser.

    Example
    -------
    Use the CLI for easiest use. Run the following to see options.
    ```
    python main.py --help
    ```
    """
    args = MyArgs().parse_args()
    save_console_outputs("console_outputs.log")
    print(args)

    if args.download:
        download_and_unzip(DATA_SOURCE_URLS, args.download_dst)

    if args.process_for_torchgeo:
        make_torchgeo_friendly(args.download_dst, args.georef_and_timestamp_dst)

    if args.make_info_plots:
        save_examples_and_info_plots(
            img_dir=args.examples_img_src,
            lbl_dir=args.examples_lbl_src,
            figures_dir=args.figures_dir,
            n_batches=10,
        )

    print("Finished!")


def download_and_unzip(data_source_urls, dst):
    for name, src in data_source_urls.items():
        dst_dir = dst / name
        print(f"Downloading {src} to {dst_dir} ...")
        to_unzip = download_file(src, dst_dir)
        print(f"Downloaded {src} to {dst_dir}!")
        print(f"Unzipping {dst_dir} ...")
        with py7zr.SevenZipFile(to_unzip, mode="r") as z:
            z.extractall(dst)
        print(f"Unzipped {dst_dir}!")


def make_torchgeo_friendly(src_root, dst_root):
    """Creates a new dataset that works with torchgeo.

    Tasks:
        1. Takes georef data from images and adds to labels (transform and crs)
        2. Adds pseudo timestamps to not overlap everything taken at the same location

    """
    assert src_root.exists()

    im_lab_pairs = (
        (datetime(2050, 1, 1, 0, 0, 0), "Oil", "Mask_oil"),
        (datetime(2060, 1, 1, 0, 0, 0), "No_oil", "Mask_no_oil"),
        (datetime(2070, 1, 1, 0, 0, 0), "Lookalike", "Mask_lookalike"),
        (datetime(2080, 1, 1, 0, 0, 0), "Test_images/Oil", "Mask_test_images/Oil"),
        (
            datetime(2090, 1, 1, 0, 0, 0),
            "Test_images/No oil",
            "Mask_test_images/No oil",
        ),
        (
            datetime(2100, 1, 1, 0, 0, 0),
            "Test_images/Lookalike",
            "Mask_test_images/Lookalike",
        ),
    )

    for base_time, img_dir, lbl_dir in im_lab_pairs:
        print(
            f"{src_root / img_dir} -> {dst_root / (img_dir + '_timestamped')} "
            "and {src_root / lbl_dir} -> "
            "{dst_root / (lbl_dir + '_georef_timestamped')} ... "
        )
        georeference_and_timestamp_images_and_masks(
            root_dir=src_root,
            dst_dir=dst_root,
            image_dir=img_dir,
            mask_dir=lbl_dir,
            output_image_dir=img_dir + "_timestamped",
            output_mask_dir=lbl_dir + "_georef_timestamped",
            base_time=base_time,
        )
        print(
            f"Finished with {img_dir + '_timestamped'} and "
            " {lbl_dir + '_georef_timestamped'}. "
        )


def save_examples_and_info_plots(
    img_dir, lbl_dir, figures_dir: Path, n_batches: int = 10
):
    os.makedirs(figures_dir, exist_ok=True)

    img_ds = TtomImageDataset(img_dir)
    lbl_ds = TtomLabelDataset(lbl_dir)

    ds = IntersectionDataset(
        dataset1=img_ds,
        dataset2=lbl_ds,
        collate_fn=concat_samples,
    )

    # samp = GridGeoSampler(ds, (512, 512), (512, 512))
    
    integral_mask, integral_transform = build_integral_mask_from_raster_dataset(
        lbl_ds
    )
    samp = BalancedRandomGeoSampler(
        ds, 
        256, 
        pos_ratio=0.5,
        integral_mask=integral_mask,
        integral_transform=integral_transform,
    )

    dl = DataLoader(
        ds,
        sampler=samp,
        batch_size=4,
        collate_fn=stack_samples,
    )

    for i, sample in enumerate(dl):
        img = sample["image"]
        mask = sample["mask"]
        info_plots(img, mask, figures_dir / str(i))
        if i >= n_batches:
            break


if __name__ == "__main__":
    main()
