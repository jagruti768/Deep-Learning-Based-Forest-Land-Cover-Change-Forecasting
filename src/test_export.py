# test_export.py - checks the GeoTIFF export using the "2023 = 2013" baseline
import numpy as np
import rasterio
from config import get_area
from export_tif import save_prediction_tif

cfg = get_area()
ds = np.load(cfg["output_dir"] / "dataset.npz")

pred = ds["lulc"][-2]                       # the 2013 map used as the "prediction"
out = cfg["output_dir"] / "baseline_pred_2023.tif"
save_prediction_tif(pred, ds, cfg, out)

# read the two files back and compare them
with rasterio.open(out) as a, rasterio.open(cfg["data_dir"] / "LULC_2023.tif") as b:
    p, t = a.read(1), b.read(1)
    print("same shape:", p.shape == t.shape)
    print("same pixel size:", a.res == b.res, a.res)
    valid = ds["valid"]
    print(f"baseline accuracy vs real 2023: {(p[valid] == t[valid]).mean() * 100:.1f}%")