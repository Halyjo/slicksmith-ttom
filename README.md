# slicksmith-ttom

```{warning}
This repo is under development and probably very buggy. If there are problems, please raise issues or just fire off PRs.
```

Processing tools for Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models.

This is code for processing and working with a a three part dataset that can be found here:

- Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C., & Monterrubio-Martínez, E. (2024). Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part I. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8346860
- Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C., & Monterrubio-Martínez, E. (2024). Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part II. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8253899
- Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C., & Monterrubio-Martínez, E. (2024). Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part III (Version 2024) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13761290


## Getting started
1. git clone <this-repo>
2. `uv sync` inside repo root folder.
3. See options by running:

```bash
❯ uv run src/slicksmith_ttom/main.py --help
usage: main.py [--download] [--process_for_torchgeo] [--make_info_plots] --download_dst DOWNLOAD_DST --georef_and_timestamp_dst GEOREF_AND_TIMESTAMP_DST
               [--figures_dir FIGURES_DIR] [-h]

options:
  --download            (bool, default=True)
  --process_for_torchgeo
                        (bool, default=False)
  --make_info_plots     (bool, default=False)
  --download_dst DOWNLOAD_DST
                        (Path, required)
  --georef_and_timestamp_dst GEOREF_AND_TIMESTAMP_DST
                        (Path, required)
  --figures_dir FIGURES_DIR
                        (Path, default=figures)
  -h, --help            show this help message and exit
```

- `--download`, `--process_for_torchgeo` and `--make_info_plots` are flags for what to do. Set all to `1` if you want to do everything.

- `--download_dst`: Path to the directory to download and unzip the files to.
- `--georef_and_timestamp_dst`: Where to store the torchgeo processed files (see [processing-for-torchgeo](#processing-for-torchgeo) for more info).
- `--make_info_plots`: When the preparation is done, it's often useful to get a better understanding of the intensity distribution and see a few examples from the dataset. If this is set to `1`, the following are generated for a small set of images `--figures_dir`:
    1. Image grids: A set of sampled images in a resolution of (480, 480) are shown. VV and VH channels are shown separately.
    2. Mask grids corresponding to the Image grids. 
    3. Histograms corresponding to the image files. Plots for the VV and the VH channels are shown separately and there are separate plots for linear and logarithmic y-scale.


### To only download (default), run:
```bash
uv run src/slicksmith_ttom/main.py --download_dst=<your-destination-data-path>
```


4. Assuming you have the processed date, the following components are good starting points to work with the data:
```python
from slicksmith_ttom import (
    TtomDataModule, ## Lightning data module with methods train_dataloader(), etc. Uses custom BalancedRandomGeoSampler 
    TtomImageDataset, ## subclass of torchgeo.datasets.RasterDataset for images only
    TtomLabelDataset, ## subclass of torchgeo.datasets.RasterDataset for labels only (used with IntersectionDataset in TtomDataModule),
    BalancedRandomGeoSampler,
    build_integral_mask_from_raster_dataset, ## to make course lookup map for faster sampling.
)
from pathlib import Path
from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset, concat_samples, stack_samples
from torchgeo.samplers import GridGeoSampler

img_data_path = Path("<your-data-root-path>/Oil_timestamped")
lbl_data_path = Path("<your-data-root-path>/Mask_oil_georef_timestamped")

img_ds = TtomImageDataset(img_dir)
lbl_ds = TtomLabelDataset(lbl_dir)

ds = IntersectionDataset(
    dataset1=img_ds,
    dataset2=lbl_ds,
    collate_fn=concat_samples,
)

## To go through the whole dataset of all images sequentially in a grid-pattern 
samp = GridGeoSampler(ds, (480, 480), (480, 480))

## Uncomment below to use the cooler sampler
## integral_mask and integral_transform are optional in BalancedRandomGeoSampler. 
## If not provided, takes less memory and is more transparent, but is much slower.

# integral_mask, integral_transform = build_integral_mask_from_raster_dataset(
#     lbl_ds
# )
# samp = BalancedRandomGeoSampler(
#     ds, 
#     size=256, 
#     pos_ratio=0.5,
#     integral_mask=integral_mask,
#     integral_transform=integral_transform,
# )

dl = DataLoader(
    ds,
    sampler=samp,
    batch_size=16,
    collate_fn=stack_samples,
)

for i, sample in enumerate(dl):
    img = sample["image"]
    mask = sample["mask"]
    print(img.shape)
    print(mask.shape)
    break

```

## Processing for `TorchGeo`

There are two things that need to be fixed to use this dataset with `Torchgeo`. (see. `src/slicksmith_ttom/preprocessing/add_georef_and_timestamps.py` for the implementation).

### 1. Timestamps

To use the TorchGeo framework effectively, our approach is to subclass `torchgeo.datasets.RasterDataset` with some reasonable class attributes and implement a sampler on top of it (see. `src/slicksmith_ttom/deep_learning/torchgeo_datasets.py`). However, to do this we need timestamps in file names that can be read given an appropriate `filename_regex` to avoid that images that overlap in space but not in time are mapped on top of each other.
I do not know the real timestamps so I just add a pseudo timestamp making sure the image and label timestamps match. 

### 2. Georeferencing Labels

To use `torchgeo.datasets.IntersectionDataset` to match images and labels (see. `src/slicksmith_ttom/deep_learning/torchgeo_datasets.py`) we need the label-tif-files to include georeference information so we copy a new set of labels in a neighboring folder with the georef info added. 

**The name:**
slicksmith-ttom: "slick": oil spill slicks, "smith": tools, "ttom": dataset author names' first characters

## References

### You can find the dataset parts here:
- Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C., & Monterrubio-Martínez, E. (2024). Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part I. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8346860
- Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C., & Monterrubio-Martínez, E. (2024). Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part II. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8253899
- Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C., & Monterrubio-Martínez, E. (2024). Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part III (Version 2024) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13761290


### Private overleaf doc 
Details for me to remember: https://www.overleaf.com/project/6812010057715ba1a6d19142
