# data_prep.py
# data_prep.py
# Reads the GeoTIFFs and converts them into ONE clean numpy file that the
# models will use. Run:  python src/data_prep.py

import sys
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use("Agg")                       # save pictures without a window
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import get_area


def round_tf(t):
    """Round an affine transform so tiny float differences don't matter."""
    return tuple(round(v, 3) for v in tuple(t)[:6])

def remap_codes(arr, mapping):
    """Convert one year's class numbers to the standard numbering.
    Values not in the mapping (e.g. nodata 15) are left unchanged."""
    if mapping is None:
        return arr
    lut = np.arange(256, dtype="uint8")        # lookup table: value -> value
    lut[:len(mapping)] = mapping               # change only the class codes
    return lut[arr]

def load_lulc(cfg):
    """Load every year's land-cover map into one array of shape (T, H, W)."""
    maps, ref = [], None
    for year in cfg["years"]:
        path = cfg["data_dir"] / cfg["lulc_pattern"].format(year=year)
        with rasterio.open(path) as src:
            arr = remap_codes(src.read(1), cfg.get("remap", {}).get(year))
            if ref is None:                  # first year defines the grid
                ref = {"transform": src.transform, "shape": src.shape,
                       "crs": CRS.from_string(cfg["crs"])}
            elif src.shape != ref["shape"] or \
                    round_tf(src.transform) != round_tf(ref["transform"]):
                raise ValueError(f"{path.name} is not on the same grid as the first year")
        maps.append(arr)
        print(f"loaded {path.name}  shape={arr.shape}")
    return np.stack(maps), ref


def load_aux(cfg, ref):
    """Load slope/elevation, move onto the land-cover grid, scale to 0..1."""
    layers, names = [], []
    for name, fname in cfg["aux"].items():
        with rasterio.open(cfg["data_dir"] / fname) as src:
            data = src.read(1).astype("float32")
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            out = np.full(ref["shape"], np.nan, dtype="float32")
            # slope grid is a few metres shifted -> resample onto LULC grid
            reproject(source=data, destination=out,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref["transform"], dst_crs=ref["crs"],
                      resampling=Resampling.nearest,
                      src_nodata=np.nan, dst_nodata=np.nan)
        lo, hi = np.nanmin(out), np.nanmax(out)
        out = (out - lo) / (hi - lo)                 # min-max scale to 0..1
        out[np.isnan(out)] = np.nanmedian(out)       # fill the few gaps
        layers.append(out)
        names.append(name)
        print(f"loaded aux '{name}': original range {lo:.0f}..{hi:.0f} -> scaled 0..1")
    return np.stack(layers).astype("float32"), names


def summary(lulc, valid, cfg):
    """Print class percentages per year and how much changed 2013 -> 2023."""
    classes = cfg["classes"]
    n = len(classes)
    print("\n=== Class share (% of valid pixels) ===")
    print("year  " + "  ".join(f"{classes[c][:10]:>10}" for c in classes))
    for t, year in enumerate(cfg["years"]):
        counts = np.bincount(lulc[t][valid], minlength=n)[:n]
        pct = 100 * counts / counts.sum()
        print(f"{year}  " + "  ".join(f"{p:10.1f}" for p in pct))

    prev, last = lulc[-2][valid].astype(int), lulc[-1][valid].astype(int)
    changed = (prev != last).mean() * 100
    print(f"\nPixels that changed {cfg['years'][-2]} -> {cfg['years'][-1]}: {changed:.1f}%")
    print(f"=> 'no change' baseline accuracy = {100 - changed:.1f}%  "
          f"(a real model must beat this)")
    print("\nTransition matrix (rows=from, cols=to), % of valid pixels:")
    tm = np.bincount(prev * n + last, minlength=n * n).reshape(n, n)
    print(np.round(100 * tm / tm.sum(), 1))


def plot_overview(lulc, aux, valid, cfg):
    classes, colors = cfg["classes"], cfg["colors"]
    cmap = ListedColormap([colors[c] for c in sorted(classes)])
    T = len(cfg["years"])
    fig, axes = plt.subplots(1, T + 1, figsize=(4 * (T + 1), 5))
    for t, ax in enumerate(axes[:T]):
        ax.imshow(np.ma.masked_where(~valid, lulc[t]), cmap=cmap,
                  vmin=0, vmax=len(classes) - 1, interpolation="nearest")
        ax.set_title(str(cfg["years"][t])); ax.axis("off")
    axes[T].imshow(np.ma.masked_where(~valid, aux[0]), cmap="terrain")
    axes[T].set_title("slope (scaled)"); axes[T].axis("off")
    plt.tight_layout()
    path = cfg["output_dir"] / "overview.png"
    plt.savefig(path, dpi=100)
    print(f"saved {path}")


def main():
    area = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = get_area(area)
    print(f"Study area: {cfg['name']}")

    lulc, ref = load_lulc(cfg)
    aux, aux_names = load_aux(cfg, ref)

    # a pixel is valid only if it is inside the park in EVERY year
    valid = np.all(lulc != cfg["nodata"], axis=0)
    print(f"\nvalid pixels: {valid.sum():,} of {valid.size:,}")

    # safety check: only known class values inside the valid area
    found = set(np.unique(lulc[:, valid]).tolist())
    unknown = found - set(cfg["classes"])
    if unknown:
        raise ValueError(f"Unexpected class values {unknown}; fix 'classes' in config.py")

    summary(lulc, valid, cfg)
    plot_overview(lulc, aux, valid, cfg)

    out = cfg["output_dir"] / "dataset.npz"
    np.savez_compressed(
        out,
        lulc=lulc.astype("uint8"),           # (T, H, W)
        aux=aux,                             # (C, H, W) scaled 0..1
        valid=valid,                         # (H, W) True inside the forest
        years=np.array(cfg["years"]),
        aux_names=np.array(aux_names),
        transform=np.array(tuple(ref["transform"])[:6]),
        crs_wkt=np.array(ref["crs"].to_wkt()),
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()