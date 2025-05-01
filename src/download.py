import os
from pathlib import Path
import requests
import py7zr
from utils import save_outputs


def download_file(url, dst_dir, filename=None, chunk_size=8192):
    os.makedirs(dst_dir, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1]

    dst_path = os.path.join(dst_dir, filename)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    return dst_path


if __name__ == "__main__":
    download_urls = dict(
        train_val_masks="https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_mask.7z",
        train_val_lookalike_mask="https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_mask.7z",
        train_val_images="https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_images.7z",
        train_val_lookalike_images="https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_images.7z",
        train_val_no_oil_images="https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_Images.7z",
        train_val_no_oil_mask="https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_mask.7z",
        test_images_masks="https://zenodo.org/records/13761290/files/02_Test_images_and_ground_truth.7z",
    )

    dst = Path("/storage/experiments/data/Trujillo/")

    save_outputs(dst / "outputs.log")

    for name, src in download_urls.items():
        dst_dir = dst / name
        print(f"Downloading {src} to {dst_dir} ...")
        to_unzip = download_file(src, dst_dir)
        print(f"Downloaded {src} to {dst_dir}!")
        print(f"Unzipping {dst_dir} ...")
        with py7zr.SevenZipFile(to_unzip, mode="r") as z:
            z.extractall(dst)
        print(f"Unzipped {dst_dir}!")
