# export_tif.py
# Turns a predicted class map (a numpy array) into a proper GeoTIFF that has
# the same location/grid/projection as your input maps. Used for 2023 now
# and for 2030 etc. later.

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


def hex_to_rgba(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def save_prediction_tif(pred, dataset, cfg, out_path):
    """
    pred    : (H, W) array of predicted class numbers (0..4 for SGNP)
    dataset : the loaded dataset.npz (gives us valid mask, transform, crs)
    cfg     : area settings from config.py (nodata value, class colours)
    out_path: where to save the .tif
    """
    # outside the park -> write the nodata value (15), same as your inputs
    out = np.where(dataset["valid"], pred, cfg["nodata"]).astype("uint8")

    transform = Affine(*dataset["transform"])      # pixel -> map coordinates
    crs = CRS.from_wkt(str(dataset["crs_wkt"]))    # EPSG:32643

    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=out.shape[0], width=out.shape[1], count=1, dtype="uint8",
        crs=crs, transform=transform, nodata=cfg["nodata"], compress="lzw",
    ) as dst:
        dst.write(out, 1)
        # embed colours so the map looks right in QGIS/ArcGIS
        cmap = {c: hex_to_rgba(col) for c, col in cfg["colors"].items()}
        cmap[cfg["nodata"]] = (0, 0, 0, 0)         # transparent outside park
        dst.write_colormap(1, cmap)
    print(f"saved {out_path}")