import os

import requests


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
