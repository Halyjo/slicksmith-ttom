# slicksmith-ttom
Processing tools for Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models.

This is code for processing and working with a a three part dataset that can be found here:

- Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C., & Monterrubio-Martínez, E. (2024). Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part I. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8346860
- Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C., & Monterrubio-Martínez, E. (2024). Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part II. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8253899
- Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C., & Monterrubio-Martínez, E. (2024). Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part III (Version 2024) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13761290


## Getting started
1. git clone <this-repo>
2. `uv sync` inside repo root folder.
3. For download, unzip and processing options, run:

```bash
uv run python -c "from slicksmith_ttom import main; main()" --help
```
Remove the `--help`-flag when you are ready to run things. You will need to specify the path to the destination folder for the download (download_dst), the destination folder for the torchgeo friendly processed data (georef_and_timestamped_dst) and a folder for the info plots and figures to go (figures_dir). There are optional flags to opt out of any of the three steps as well. 

**eg. if you only want to download and unzip, run:**
```bash
uv run python -c "from slicksmith_ttom import main; main()" --process_for_torchgeo=0 --make_info_plots=0
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
samp = GridGeoSampler(ds, (512, 512), (512, 512))


## Uncomment below to use the cooler sampler
## integral_mask and integral_transform are optional in BalancedRandomGeoSampler. 
## If not provided, takes less memory, but is much slower.

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


**The name:**
slicksmith-ttom: "slick": oil spill slicks, "smith": tools, "ttom": dataset author names' first characters

## References
Private overleaf doc with some details for me to remember: https://www.overleaf.com/project/6812010057715ba1a6d19142
