# run_baselines.py - persistence baseline + Random Forest
# Inputs: 1993, 2003, 2013 (+ slope)  ->  predict 2023  ->  compare with real 2023
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config import get_area
from splits import block_split
from features import build_features
from metrics import evaluate, print_report, area_table
from export_tif import save_prediction_tif
from plots import save_comparison_figure


def main():
    cfg = get_area()
    ds = np.load(cfg["output_dir"] / "dataset.npz")
    lulc, aux, valid = ds["lulc"], ds["aux"], ds["valid"]
    names = [cfg["classes"][c] for c in sorted(cfg["classes"])]
    n = len(names)

    train, val, test = block_split(valid)
    print(f"train pixels {train.sum():,} | val {val.sum():,} | test {test.sum():,}")

    inputs = lulc[:-1]                    # 1993, 2003, 2013
    target = lulc[-1].astype(int)         # real 2023
    prev = lulc[-2].astype(int)           # 2013

    # ---------- Model 0: persistence ("nothing changes") ----------
    r0 = evaluate(target[test], prev[test], prev[test], n)
    print_report("Persistence baseline (2023 = 2013), TEST blocks", r0, names)

    # ---------- Model 1: Random Forest ----------
    print("\nBuilding features ...")
    X = build_features(inputs, aux, n)                    # (H, W, F)
    Xf = X.reshape(-1, X.shape[-1])
    yf = target.ravel()

    rng = np.random.default_rng(0)
    idx = np.flatnonzero(train.ravel())
    idx = rng.choice(idx, size=min(300_000, idx.size), replace=False)

    print(f"Training Random Forest on {idx.size:,} pixels (a few minutes) ...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_leaf=5,
                                n_jobs=-1, random_state=0)
    rf.fit(Xf[idx], yf[idx])

    print("Predicting the whole map ...")
    out = np.zeros(valid.size, dtype=np.uint8)
    vidx = np.flatnonzero(valid.ravel())
    for s in range(0, vidx.size, 500_000):                # in chunks to save memory
        chunk = vidx[s:s + 500_000]
        out[chunk] = rf.predict(Xf[chunk])
    pred = out.reshape(valid.shape)

    r1 = evaluate(target[test], pred[test], prev[test], n)
    print_report("Random Forest, TEST blocks (never seen in training)", r1, names)

    print("\nArea per class on the TEST blocks (km2):")
    area_table(target[test], pred[test], names)

    # how much did the model rely on history vs slope? (links to the 80/20 idea)
    imp = rf.feature_importances_
    n_hist = (inputs.shape[0] + 1) * n          # year one-hots + neighbourhood fractions
    print(f"\nRF feature importance: land-cover history {100 * imp[:n_hist].sum():.1f}% | "
          f"slope/aux {100 * imp[n_hist:].sum():.1f}%")

    # ---------- outputs ----------
    np.save(cfg["output_dir"] / "pred_2023_rf.npy", pred)
    save_prediction_tif(pred, ds, cfg, cfg["output_dir"] / "pred_2023_rf.tif")
    save_comparison_figure(target, pred, valid, test, cfg,
                           cfg["output_dir"] / "rf_vs_actual_2023.png",
                           "Random Forest: predicted 2023 vs actual 2023")


if __name__ == "__main__":
    main()