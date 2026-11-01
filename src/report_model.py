# report_model.py - full per-class report of an already-trained model
# usage:  python src/report_model.py unet_w80
import sys
import numpy as np
from config import get_area
from splits import block_split
from metrics import evaluate, print_report, area_table

tag = sys.argv[1]
cfg = get_area()
out = cfg["output_dir"]
ds = np.load(out / "dataset.npz")
lulc, valid = ds["lulc"], ds["valid"]
names = [cfg["classes"][c] for c in sorted(cfg["classes"])]
_, _, test = block_split(valid)

target = lulc[-1].astype(int)
prev = lulc[-2].astype(int)
pred = np.load(out / f"pred_2023_{tag}.npy").astype(int)

r = evaluate(target[test], pred[test], prev[test], len(names))
print_report(f"{tag} (TEST blocks)", r, names)
print("\nArea per class on the TEST blocks (km2):")
area_table(target[test], pred[test], names)