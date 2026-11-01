# inspect_tif.py   usage:  python src/inspect_tif.py data/sgnp/LULC_1993.tif
import sys
import numpy as np
import rasterio

with rasterio.open(sys.argv[1]) as s:
    print("size", s.width, "x", s.height, "| crs", s.crs, "| pixel", s.res)
    a = s.read(1)
    try:
        cm = s.colormap(1)
    except Exception:
        cm = {}
    for v, c in zip(*np.unique(a, return_counts=True)):
        print(f"value {v:3d}   pixels {c:10,d}   colour(RGBA) {cm.get(int(v), '-')}")