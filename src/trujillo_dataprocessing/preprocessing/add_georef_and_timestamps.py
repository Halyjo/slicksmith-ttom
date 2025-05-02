import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import rasterio
from tqdm import tqdm


def make_timestamped_filename(index: int, base_time) -> str:
    """Make artificial timestamp for torchgeo not to
    overlay images from same location at different times.
    Make sure to use a non sensical time so no one will think they are real.
    """
    dt = base_time + timedelta(seconds=index)
    return f"{index:05d}_{dt.strftime('%Y%m%dT%H%M%S')}.tif"


def georeference_and_timestamp_images_and_masks(
    root_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    image_dir="Oil",
    mask_dir="Mask_oil",
    output_image_dir="Oil_timestamped",
    output_mask_dir="Mask_oil_georef_timestamped",
    base_time=datetime(2050, 1, 1, 0, 0, 0),
):
    """
    Add geospatial metadata to oil spill mask files by copying it from the
    corresponding Sentinel-1 image.
    Rename both image and mask files to include artificial
    timestamps (set in year 2050).
    This is to avoid implying real acquisition times and
    ensures compatibility with TorchGeo's temporal logic.
    """
    root_dir = Path(root_dir)

    os.makedirs(dst_dir / output_image_dir, exist_ok=True)
    os.makedirs(dst_dir / output_mask_dir, exist_ok=True)

    image_filenames = sorted(
        f for f in os.listdir(root_dir / image_dir) if f.endswith(".tif")
    )

    for filename in tqdm(image_filenames, desc="Processing image-mask pairs"):
        image_path = root_dir / image_dir / filename
        mask_path = root_dir / mask_dir / filename
        ## In Mask_test_images, files look like: 00000_segmentation.tif
        index = int(Path(filename).stem.split("_")[0])
        new_filename = make_timestamped_filename(index, base_time)
        output_image_path = dst_dir / output_image_dir / new_filename
        output_mask_path = dst_dir / output_mask_dir / new_filename

        # Read image data and metadata
        with rasterio.open(image_path) as img_src:
            image_data = img_src.read()
            crs = img_src.crs
            transform = img_src.transform
            img_dtype = img_src.dtypes[0]
            count = img_src.count
            height = img_src.height
            width = img_src.width

        # Write the image with same metadata and new filename
        with rasterio.open(
            output_image_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=img_dtype,
            crs=crs,
            transform=transform,
        ) as dst_img:
            dst_img.write(image_data)

        # Read the corresponding mask
        try:
            with rasterio.open(mask_path) as mask_src:
                mask_data = mask_src.read(1)  # single-channel
                mask_dtype = mask_src.dtypes[0]
        except Exception:
            with rasterio.open(
                str(mask_path).replace(".tif", "_segmentation.tif")
            ) as mask_src:
                mask_data = mask_src.read(1)  # single-channel
                mask_dtype = mask_src.dtypes[0]

        # Write the mask with geospatial metadata and artificial timestamp tag
        timestamp = base_time + timedelta(seconds=index)
        tags = {"TIFFTAG_DATETIME": timestamp.strftime("%Y:%m:%d %H:%M:%S")}

        with rasterio.open(
            output_mask_path,
            "w",
            driver="GTiff",
            height=mask_data.shape[0],
            width=mask_data.shape[1],
            count=1,
            dtype=mask_dtype,
            crs=crs,
            transform=transform,
        ) as dst_mask:
            dst_mask.write(mask_data, 1)
            dst_mask.update_tags(**tags)

    print(f"All timestamped images saved to: {dst_dir / output_image_dir}")
    print(f"All georeferenced masks saved to: {dst_dir / output_mask_dir}")
