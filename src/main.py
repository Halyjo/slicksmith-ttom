from src.download import download_file
from preprocessing.add_georef_and_timestamps import (
    georeference_and_timestamp_images_and_masks,
)
from pathlib import Path
from utils import save_outputs
import py7zr
import datetime


DATA_SOURCE_URLS = dict(
    train_val_masks="https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_mask.7z",
    train_val_lookalike_mask="https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_mask.7z",
    train_val_images="https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_images.7z",
    train_val_lookalike_images="https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_images.7z",
    train_val_no_oil_images="https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_Images.7z",
    train_val_no_oil_mask="https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_mask.7z",
    test_images_masks="https://zenodo.org/records/13761290/files/02_Test_images_and_ground_truth.7z",
)


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
    # root_dir = Path("/storage/experiments/data/Trujillo")
    dst_root = Path("/storage/experiments/data/Trujillo_torchgeo")
    save_outputs(dst_root / "add_georef_timestamp.log")

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
            f"{img_dir} -> {img_dir + '_timestamped'} and {lbl_dir} -> {lbl_dir + '_georef_timestamped'} ... "
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
            f"Finished with {img_dir + '_timestamped'} and {lbl_dir + '_georef_timestamped'}. "
        )


def main():
    download_dst = Path("/storage/experiments/data/Trujillo/")
    processed_data_dst = Path("/storage/experiments/data/Trujillo_torchgeo/")
    save_outputs(download_dst / "outputs.log")

    download_and_unzip(DATA_SOURCE_URLS, download_dst)
    make_torchgeo_friendly(download_dst, processed_data_dst)


if __name__ == "__main__":
    main()
