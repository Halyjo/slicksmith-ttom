# %%
from pathlib import Path

from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler
from utils import view_geodataloader_on_map

from slicksmith_ttom.deep_learning import (
    BalancedRandomGeoSampler,
    TtomImageDataset,
    TtomLabelDataset,
    build_integral_mask_from_raster_dataset,
)


# %%
def main():
    """Illustrations in folium

    TODO:
    1. [x] Load the label dataset TtomLabelDataset with two naive samplers: random and grid.
    2. [x] Make a folium map with a box showing the total bounds of the label dataset.
    3. [ ] Load 10 bounding boxes of each and plot on a folium map as boxes. Use one color for the grid sampled boxes
        and another for the random sampled.
    4. [ ] Make an integral mask for the BalancedRandomGeoSampler and show that image on top of the map.
    5. [ ] Sample a set of boxes with the BalancedRandomGeoSampler and show those in a third color.

    """
    pass
    # %%
    label_root = Path("/Users/hjo109/Documents/data/Ttom/Mask_oil_georef_timestamped")
    image_root = Path("/Users/hjo109/Documents/data/Ttom/Oil_timestamped")
    n_images = 16
    img_size = (1024, 1024)
    label_dataset = TtomLabelDataset(label_root)
    image_dataset = TtomImageDataset(image_root)
    seg_dataset = IntersectionDataset(
        dataset1=image_dataset,
        dataset2=label_dataset,
    )
    # %%
    ## Grid
    grid_sampler = GridGeoSampler(seg_dataset, size=img_size, stride=img_size)
    # grid_sampler = RandomBatchGeoSampler(seg_da, size=img_size, stride=img_size)
    grid_loader = DataLoader(
        seg_dataset,
        batch_size=1,
        sampler=grid_sampler,
        collate_fn=stack_samples,
    )
    sample = next(iter(grid_loader))
    print(sample)
    # breakpoint()

    m = view_geodataloader_on_map(grid_loader, maxnum=20, show_images=False)
    m

    # %%

    # %%
    ## Random
    rand_sampler = RandomGeoSampler(label_dataset, size=img_size, length=500)
    rand_loader = DataLoader(
        seg_dataset,
        batch_size=1,
        sampler=rand_sampler,
        collate_fn=stack_samples,
    )
    m = view_geodataloader_on_map(rand_loader, maxnum=10, show_images=True)
    m
    # %%

    # %%

    ## Balanced Random
    integral_mask, integral_transform, mosaic, bounds = (
        build_integral_mask_from_raster_dataset(label_dataset, return_mosaic_raw=True)
    )

    binary_mask = (integral_mask > 0).cpu().numpy()  # (H, W) uint8/bool is fine
    # src_crs      = label_ds.crs                     # CRS you mosaicked into

    balanced_sampler = BalancedRandomGeoSampler(
        seg_dataset,
        size=img_size,
        length=10,
        pos_ratio=1.0,
        # integral_mask=integral_mask,
        # integral_transform=integral_transform,
    )
    balanced_loader = DataLoader(
        seg_dataset,
        batch_size=1,
        sampler=balanced_sampler,
        collate_fn=stack_samples,
    )
    for sample in balanced_loader:
        if sample["mask"].sum() == 0:
            print(".", end="")
            continue

    from torchshow import torchshow as ts

    ts.show(sample["image"])
    ts.show(sample["mask"])
    print(sample)
    # m = view_geodataloader_on_map(balanced_loader, maxnum=16, m=m)
    # m


# %%

# %%

# %%
if __name__ == "__main__":
    main()

# %%
