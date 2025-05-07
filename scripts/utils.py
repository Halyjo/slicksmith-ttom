# %%
from collections import defaultdict
from datetime import datetime

import folium
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from torchgeo.datasets.utils import BoundingBox
from tqdm import tqdm


def to_latlon_bounds(bbox: BoundingBox, src_crs, nested=False, densify=21):
    """
    Convert torchgeo.datasets.utils.BoundingBox in `src_crs` to WGS‑84.

    Parameters
    ----------
    bbox : torchgeo.datasets.utils.BoundingBox

    src_crs : rasterio.crs.CRS or anything accepted by rasterio
    nested : bool, optional
        If True, return [[south, west], [north, east]] for Folium.
    densify : int, optional
        Number of points per edge sent to `transform_bounds`.

    Returns
    -------
    tuple | list
        west, south, east, north  *or*  [[south, west], [north, east]]
    """
    left, bottom, right, top = bbox
    west, south, east, north = transform_bounds(
        CRS.from_user_input(src_crs),
        CRS.from_epsg(4326),
        left,
        bottom,
        right,
        top,
        densify_pts=densify,
    )
    return [[south, west], [north, east]] if nested else (west, south, east, north)


def sanity_check_bounds(bbox, crs):
    w, s, e, n = to_latlon_bounds(bbox, crs)
    assert -180 <= w <= 180 and -90 <= s <= 90
    assert w < e and s < n


# %%
def get_folium_bounds(bbox, dst_crs=None, src_crs=None, nested=False, densify_pts=21):
    left = bbox.minx
    bottom = bbox.miny
    right = bbox.maxx
    top = bbox.maxy
    if dst_crs is not None:
        # dst_crs = rasterio.crs.CRS.from_epsg(4326)
        left, bottom, right, top = rasterio.warp.transform_bounds(
            src_src=src_crs,
            dst_crs=dst_crs,
            left=bbox.minx,
            bottom=bbox.miny,
            right=bbox.maxx,
            top=bbox.maxy,
            densify_pts=densify_pts,  ## More than just the corners, default is 21
        )
    if nested:
        return [[bottom, left], [top, right]]
    else:
        return left, bottom, right, top


# %%
def view_geodataloader_on_map(dataloader, maxnum, m=None, show_images=False):
    known_boxes = defaultdict(list)
    time_format = "%Y%m%d_%H:%M:%S"
    ds = dataloader.dataset
    minlong, minlat, maxlong, maxlat = get_folium_bounds(ds.bounds, nested=False)
    centerlong = (minlong + maxlong) / 2
    centerlat = (minlat + maxlat) / 2

    ## total time delta
    mint_tot = datetime.fromtimestamp(ds.bounds.mint).strftime(time_format)
    if m is None:
        m = folium.Map(location=(centerlat, centerlong), zoom_start=3)

    folium.Rectangle(
        bounds=get_folium_bounds(ds.bounds, nested=True),
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.1,
        weight=3,
        # popup=f"{mint_tot}",
        # tooltip=f"{mint_tot}",
        # tooltip="<strong>Total area</strong>",
    ).add_to(m)

    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i > maxnum:
            break

        # label = sample["mask"][0][0].numpy()
        bounds = sample["bounds"][0]
        known_boxes[bounds].append(i)

        if show_images:
            image = sample["image"][0][1].numpy()
            rgb = np.stack(
                [
                    image,
                    image,
                    image,
                ],
                axis=2,
            )

            folium.raster_layers.ImageOverlay(
                name="sar",
                image=rgb,
                bounds=get_folium_bounds(bounds, nested=True),
                interactive=True,
                tags=["sar"],
            ).add_to(m)
        mints = datetime.fromtimestamp(bounds.mint).strftime(time_format)
        # mints = datetime.utcfromtimestamp(bounds.mint).strftime(time_format)
        # datetime.utcfromtimestamp(bounds.maxt).strftime(time_format)

        rkw = {
            "color": "blue",
            "line_cap": "round",
            "fill": False,
            "weight": 5.5,
            # "popup": f"{mints}",
            "tooltip": f"{mints}",
            # "tooltip": f"<strong>Patch {i}</strong>",
        }

        # src_crs = sample["crs"][0]
        folium.Rectangle(
            bounds=get_folium_bounds(bounds, nested=True),
            tags=["rect"],
            **rkw,
        ).add_to(m)

    # if show_images:
    #     TagFilterButton(["sar", "rect"]).add_to(m)

    ## add lonlat info on click
    m.add_child(
        folium.LatLngPopup()
        # folium.ClickForLatLng(format_str='"[" + lat + "," + lng + "]"', alert=True)
    )

    if any([len(v) > 1 for v in known_boxes.values()]):
        raise ValueError(f"Found duplicate boxes: {known_boxes}")
    return m


if __name__ == "__main__":
    pass
    # m = view_geodataloader_on_map(dl, maxnum=10)
    # m  # noqa: B018

    # m.save("val_in_roi_60m_bounds.html")
    # #%%
    # m.save("all_roi_60m_bounds.html")
    #     # m.save("prechipped_all_60m_imgs_locally.html")
    #     # m.save("/storage/outputs/scaling-waddle/all_60m_patches_without_labels.html")

    # ## Find duplicate files in a directory
    # import os
    # from collections import defaultdict

    # def find_duplicates(directory):
    #     files = defaultdict(list)
    #     for file in os.listdir(directory):
    #         files[os.path.getsize(os.path.join(directory, file))].append(file)
    #         # files[file].append(file)
    #     return {size: files for size, files in files.items() if len(files) > 1}

    # find_duplicates("/Users/hjo109/Documents/data/NR_KSAT_data_subset/img_60m/test/")


# %%
