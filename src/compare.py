# compare.py - put every model's 2023 prediction side by side and pick the best
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_area
from splits import block_split
from metrics import evaluate


def main():
    cfg = get_area()
    out = cfg["output_dir"]
    ds = np.load(out / "dataset.npz")
    lulc, valid = ds["lulc"], ds["valid"]
    n = len(cfg["classes"])
    _, _, test = block_split(valid)                  # same test blocks as in training
    target = lulc[-1].astype(int)
    prev = lulc[-2].astype(int)

    preds = {"persistence": prev}
    for f in sorted(out.glob("pred_2023_*.npy")):    # every model saved by train.py
        preds[f.stem.replace("pred_2023_", "")] = np.load(f).astype(int)

    rows = []
    for name, p in preds.items():
        r = evaluate(target[test], p[test], prev[test], n)
        info_file = out / f"info_{name}.json"
        info = json.load(open(info_file)) if info_file.exists() else {}
        rows.append((name, r, info))
    rows.sort(key=lambda t: -t[1]["fom"])

    print(f"\nTEST blocks only ({test.sum():,} pixels)")
    print(f"{'model':<24}{'OA %':>7}{'Kappa':>8}{'macroF1':>9}{'change %':>10}{'FoM %':>8}{'params':>10}{'min':>7}")
    for name, r, e in rows:
        print(f"{name:<24}{100 * r['oa']:>7.2f}{r['kappa']:>8.3f}{r['macro_f1']:>9.3f}"
              f"{100 * r['change_acc']:>10.2f}{100 * r['fom']:>8.2f}"
              f"{e.get('params', '-'):>10}{e.get('minutes', '-'):>7}")

    models = [t for t in rows if t[0] != "persistence"]
    if models:
        print("\nBest model by each score:")
        for key, label in (("oa", "Overall accuracy"), ("kappa", "Kappa"),
                           ("macro_f1", "Macro F1"), ("change_acc", "Change-only accuracy"),
                           ("fom", "Figure of Merit")):
            best = max(models, key=lambda t: t[1][key])
            print(f"  {label:<22}: {best[0]}")

    # save table + chart
    with open(out / "model_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "OA", "kappa", "macro_f1", "change_acc", "fom"])
        for name, r, _ in rows:
            w.writerow([name, r["oa"], r["kappa"], r["macro_f1"], r["change_acc"], r["fom"]])

    labels = [t[0] for t in rows]
    scores = [("Overall acc", "oa"), ("Macro F1", "macro_f1"),
              ("Change-only acc", "change_acc"), ("FoM", "fom")]
    x = np.arange(len(labels))
    wd = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (lab, key) in enumerate(scores):
        ax.bar(x + (i - 1.5) * wd, [100 * t[1][key] for t in rows], wd, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("%")
    ax.set_title("Predicting 2023 from 1993/2003/2013 - TEST blocks")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", ncol=4)
    plt.tight_layout()
    plt.savefig(out / "model_comparison.png", dpi=110)
    print(f"\nsaved {out / 'model_comparison.csv'} and model_comparison.png")


if __name__ == "__main__":
    main()