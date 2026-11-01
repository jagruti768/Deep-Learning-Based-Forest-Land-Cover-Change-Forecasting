# make_report_tables.py
# Journal-style tables:
#   Table A - area (km2 and %) of every class in every year (+ the forecast year)
#   Table B - accuracy of the model's prediction of the last real year
#             (user's accuracy = precision, producer's accuracy = recall, OA, kappa)
# usage:  python src/make_report_tables.py                  (area + model from config.py)
#         python src/make_report_tables.py sgnp unet_w80
import csv
import sys

import numpy as np
import rasterio

from config import get_area
from splits import block_split
from metrics import evaluate


def show(rows):
    """Print a table with aligned columns."""
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for k, r in enumerate(rows):
        print("  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i])
                        for i, c in enumerate(r)))
        if k == 0:
            print("-" * (sum(w) + 2 * (len(w) - 1)))


def save(rows, path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"saved {path}")


def main():
    area = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = get_area(area)
    tag = sys.argv[2] if len(sys.argv) > 2 else cfg.get("best_model", "unet_w80")
    out = cfg["output_dir"]

    ds = np.load(out / "dataset.npz")
    lulc, valid = ds["lulc"], ds["valid"]
    years = [int(y) for y in ds["years"]]
    n = len(cfg["classes"])
    names = [cfg["classes"][c] for c in sorted(cfg["classes"])]
    px_km2 = abs(ds["transform"][0] * ds["transform"][4]) / 1e6   # 30 m x 30 m = 0.0009

    # ------------------------------------------------ Table A: areas
    step = years[-1] - years[-2]
    target_year = years[-1] + step
    cols = [(str(y), lulc[t]) for t, y in enumerate(years)]
    fut = out / f"future_{target_year}_{tag}.tif"
    if fut.exists():
        with rasterio.open(fut) as src:
            cols.append((f"{target_year} (forecast)", src.read(1)))
    else:
        print(f"(no forecast column: {fut.name} not found - run predict_future.py {tag})")

    km2 = {label: np.bincount(arr[valid].astype(int), minlength=n)[:n] * px_km2
           for label, arr in cols}
    head = ["LULC class"]
    for label, _ in cols:
        head += [f"{label} km2", f"{label} %"]
    table_a = [head]
    for i, name in enumerate(names):
        row = [name]
        for label, _ in cols:
            row += [f"{km2[label][i]:.2f}", f"{100 * km2[label][i] / km2[label].sum():.2f}"]
        table_a.append(row)
    total = ["Total"]
    for label, _ in cols:
        total += [f"{km2[label].sum():.2f}", "100"]
    table_a.append(total)

    print("\nTable A - area of each LULC class")
    show(table_a)
    save(table_a, out / "report_table_areas.csv")

    # ------------------------------------------------ Table B: forecast accuracy
    _, _, test = block_split(valid)
    target = lulc[-1].astype(int)
    prev = lulc[-2].astype(int)
    models = [("No change (persistence)", prev)]
    pred_file = out / f"pred_2023_{tag}.npy"        # file name keeps the "2023" label
    if pred_file.exists():
        models.append((tag, np.load(pred_file).astype(int)))

    table_b = [["Accuracy (%)"] + names + ["Overall accuracy", "Kappa"]]
    for label, p in models:
        r = evaluate(target[test], p[test], prev[test], n)
        table_b.append([f"{label}: User's accuracy"]
                       + [f"{100 * v:.2f}" for v in r["precision"]]
                       + [f"{100 * r['oa']:.2f}", f"{r['kappa']:.3f}"])
        table_b.append([f"{label}: Producer's accuracy"]
                       + [f"{100 * v:.2f}" for v in r["recall"]] + ["", ""])

    print(f"\nTable B - forecast accuracy for {years[-1]} on held-out TEST blocks "
          f"({test.sum():,} pixels) vs the real {years[-1]} map")
    show(table_b)
    save(table_b, out / "report_table_accuracy.csv")


if __name__ == "__main__":
    main()