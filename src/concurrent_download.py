import os
from pathlib import Path
import asyncio
import aiohttp
import aiofiles
import py7zr
from tqdm import tqdm
from scaling_waddle.utils import save_outputs


async def download_file(session, url, dst_dir, filename=None, chunk_size=8192):
    os.makedirs(dst_dir, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1]

    dst_path = os.path.join(dst_dir, filename)

    async with session.get(url) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=filename)

        async with aiofiles.open(dst_path, "wb") as f:
            async for chunk in response.content.iter_chunked(chunk_size):
                if chunk:
                    await f.write(chunk)
                    progress.update(len(chunk))

        progress.close()

    return dst_path


def extract_7z_file(filepath, extract_to):
    with py7zr.SevenZipFile(filepath, mode="r") as z:
        z.extractall(extract_to)


async def main():
    download_urls = dict(
        # train_val_masks="https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_mask.7z",
        # train_val_lookalike_mask="https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_mask.7z",
        train_val_no_oil_mask="https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_mask.7z",
        train_val_images="https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_images.7z",
        train_val_lookalike_images="https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_images.7z",
        train_val_no_oil_images="https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_Images.7z",
        test_images_masks="https://zenodo.org/records/13761290/files/02_Test_images_and_ground_truth.7z",
    )

    dst = Path("/storage/experiments/data/Trujillo/")

    async with aiohttp.ClientSession() as session:
        download_tasks = []
        for name, url in download_urls.items():
            dst_dir = dst / name
            task = download_file(session, url, dst_dir)
            download_tasks.append(task)

        downloaded_files = await asyncio.gather(*download_tasks)

    # # Concurrent extraction
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(extract_7z_file, filepath, dst) for filepath in downloaded_files]
    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting"):
    #         future.result()

    # Simple extraction (sequential)
    for file_path in tqdm(downloaded_files, desc="Extracting"):
        extract_7z_file(file_path, dst)


if __name__ == "__main__":
    save_outputs("/storage/experiments/data/outputs.log")  # Trujillo/
    asyncio.run(main())
