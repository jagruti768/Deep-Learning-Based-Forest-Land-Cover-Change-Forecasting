# predict_future.py - use a trained model to predict one time step AHEAD
# The model learned 10-year steps, so inputs 2003, 2013, 2023 -> 2033.
# usage:  python src/predict_future.py unet_w80
import sys
import numpy as np
import torch
from config import get_area
from models import Forecaster
from dl_utils import predict_map
from export_tif import save_prediction_tif


def main():
    tag = sys.argv[1]                      # name of a file in outputs/<area>/models/
    kind = tag.split("_w")[0]              # e.g. "unet"
    lc_weight = int(tag.split("_w")[1].split("_")[0]) / 100   # e.g. 0.8

    cfg = get_area()
    out = cfg["output_dir"]
    ds = np.load(out / "dataset.npz")
    lulc, aux, valid = ds["lulc"], ds["aux"], ds["valid"]
    years = [int(y) for y in ds["years"]]
    n = len(cfg["classes"])
    names = [cfg["classes"][c] for c in sorted(cfg["classes"])]

    step = years[-1] - years[-2]           # 10 years
    target_year = years[-1] + step         # 2033
    inputs = lulc[1:]                      # the LAST three maps: 2003, 2013, 2023
    print(f"inputs {years[1:]} -> predicting {target_year}")

    model = Forecaster(kind, n, inputs.shape[0], aux.shape[0], lc_weight=lc_weight)
    model.load_state_dict(torch.load(out / "models" / f"{tag}.pt"))
    pred = predict_map(model, inputs, aux, n)

    save_prediction_tif(pred, ds, cfg, out / f"future_{target_year}_{tag}.tif")

    # area summary on the whole study area (one pixel = 0.0009 km2)
    last = np.bincount(lulc[-1][valid], minlength=n)[:n] * 0.0009
    new = np.bincount(pred[valid], minlength=n)[:n] * 0.0009
    print(f"\n{'class':<18}{years[-1]:>10}{target_year:>10}{'change %':>10}")
    for i, c in enumerate(names):
        print(f"{c:<18}{last[i]:>10.1f}{new[i]:>10.1f}{100 * (new[i] - last[i]) / last[i]:>10.1f}")


if __name__ == "__main__":
    main()