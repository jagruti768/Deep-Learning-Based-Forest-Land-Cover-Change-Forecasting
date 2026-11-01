# prepare_molai.py
# Converts the two COLOUR maps of Molai into class GeoTIFFs (class numbers 0-4, nodata 15)
# on ONE common 30 m grid:
#   data/molai/LULC_final_color.tif          2000, georeferenced colour map
#   data/molai/Molai_Forest_17_Jan_2025.tif  2025, colour PICTURE with no coordinates
# The 2025 picture is registered to the 2000 grid by cropping its island and stretching it.
# usage:  python src/prepare_molai.py
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "molai"
FILE_2000 = DATA / "LULC_final_color.tif"
FILE_2025 = DATA / "Molai_Forest_17_Jan_2025.tif"
NODATA = 15

# colour (R, G, B) -> (class number, class name); the SAME legend in both maps
LEGEND = {
    (0, 92, 230): (0, "Water bodies"),
    (255, 0, 0): (1, "Bare soil, sand, urban"),
    (115, 178, 115): (2, "Dry grasslands, shrubs"),
    (112, 168, 0): (3, "Croplands, grasslands"),
    (38, 115, 0): (4, "Forests, healthy crops"),
}


def read_rgb(path):
    with rasterio.open(path) as s:
        return np.moveaxis(s.read()[:3], 0, -1), s.transform, s.crs


def is_white(rgb):
    return np.all(rgb == 255, axis=-1)


def to_classes(rgb, name):
    """Replace each legend colour by its class number; everything else -> nodata."""
    out = np.full(rgb.shape[:2], NODATA, dtype=np.uint8)
    known = np.zeros(rgb.shape[:2], dtype=bool)
    for colour, (code, _) in LEGEND.items():
        hit = np.all(rgb == np.array(colour), axis=-1)
        out[hit] = code
        known |= hit
    odd = int((~known & ~is_white(rgb)).sum())
    print(f"{name}: {odd} pixels have a colour that is not in the legend (set to nodata)")
    return out


def write_tif(arr, path, transform, crs):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="uint8", crs=crs, transform=transform,
                       nodata=NODATA, compress="lzw") as dst:
        dst.write(arr, 1)
        cmap = {code: colour + (255,) for colour, (code, _) in LEGEND.items()}
        cmap[NODATA] = (0, 0, 0, 0)
        dst.write_colormap(1, cmap)
    print(f"saved {path}")


def main():
    rgb00, transform, crs = read_rgb(FILE_2000)
    if crs is None:
        raise SystemExit("The 2000 file has no coordinate system - cannot continue.")
    H, W = rgb00.shape[:2]
    rgb25, _, _ = read_rgb(FILE_2025)

    # ---- register the 2025 picture to the 2000 grid ----
    m25 = ~is_white(rgb25)
    rows = np.where(m25.any(axis=1))[0]
    cols = np.where(m25.any(axis=0))[0]
    crop = rgb25[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]          # just the island
    stretched = np.array(Image.fromarray(crop).resize((W, H), Image.NEAREST))

    m00, m25s = ~is_white(rgb00), ~is_white(stretched)
    iou = (m00 & m25s).sum() / (m00 | m25s).sum()
    print(f"island outline agreement between the two maps (IoU): {iou:.3f}")
    if iou < 0.95:
        raise SystemExit("The island outlines do not match well enough - check the files.")

    c00 = to_classes(rgb00, "2000")
    c25 = to_classes(stretched, "2025")
    write_tif(c00, DATA / "LULC_2000.tif", transform, crs)
    write_tif(c25, DATA / "LULC_2025.tif", transform, crs)

    # ---- quick summary ----
    names = [v[1] for v in sorted(LEGEND.values())]
    valid = (c00 != NODATA) & (c25 != NODATA)
    print(f"\nvalid pixels in both maps: {valid.sum():,}  (about {valid.sum() * 0.0009:.1f} km2)")
    print(f"{'class':<26}{'2000 %':>8}{'2025 %':>8}")
    for i, nme in enumerate(names):
        print(f"{nme:<26}{100 * (c00[valid] == i).mean():>8.1f}{100 * (c25[valid] == i).mean():>8.1f}")
    n = len(names)
    tm = np.bincount(c00[valid].astype(int) * n + c25[valid].astype(int),
                     minlength=n * n).reshape(n, n)
    print("\nTransition matrix 2000 -> 2025 (rows = 2000, cols = 2025), % of valid pixels:")
    print(np.round(100 * tm / tm.sum(), 1))
    print(f"unchanged pixels: {100 * np.trace(tm) / tm.sum():.1f}%  "
          f"(= accuracy of the 'no change' baseline)")


if __name__ == "__main__":
    main()