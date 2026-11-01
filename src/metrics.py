# metrics.py - all the scores used to compare models
import numpy as np


def confusion(y_true, y_pred, n):
    """Rows = actual class, columns = predicted class."""
    return np.bincount(y_true.astype(int) * n + y_pred.astype(int),
                       minlength=n * n).reshape(n, n)


def evaluate(y_true, y_pred, y_prev, n):
    """
    y_true: actual 2023 classes, y_pred: predicted 2023, y_prev: 2013 classes
    (all 1-D arrays holding only the pixels being scored).
    """
    cm = confusion(y_true, y_pred, n)
    total = cm.sum()
    oa = np.trace(cm) / total                                   # overall accuracy
    pe = (cm.sum(0) * cm.sum(1)).sum() / total ** 2
    kappa = (oa - pe) / (1 - pe)                                # agreement beyond chance

    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)

    # --- change-focused scores (did the model find the pixels that changed?) ---
    changed = y_true != y_prev            # pixels that REALLY changed
    pred_changed = y_pred != y_prev       # pixels the model SAYS changed
    hits = (changed & (y_pred == y_true)).sum()               # change predicted correctly
    misses = (changed & ~pred_changed).sum()                  # change not predicted
    wrong_hits = (changed & pred_changed & (y_pred != y_true)).sum()  # changed, wrong class
    false_alarms = (~changed & pred_changed).sum()            # change predicted, none happened
    fom = hits / max(hits + misses + wrong_hits + false_alarms, 1)
    change_acc = hits / max(changed.sum(), 1)

    return dict(cm=cm, oa=oa, kappa=kappa, precision=precision, recall=recall,
                f1=f1, macro_f1=f1.mean(), change_acc=change_acc, fom=fom)


def print_report(name, r, class_names):
    print(f"\n===== {name} =====")
    print(f"Overall accuracy      : {100 * r['oa']:.2f}%")
    print(f"Kappa                 : {r['kappa']:.3f}")
    print(f"Macro F1              : {r['macro_f1']:.3f}")
    print(f"Change-only accuracy  : {100 * r['change_acc']:.2f}%   (pixels that really changed)")
    print(f"Figure of Merit (FoM) : {100 * r['fom']:.2f}%")
    print(f"{'class':<18}{'precision':>10}{'recall':>10}{'F1':>8}")
    for i, c in enumerate(class_names):
        print(f"{c:<18}{r['precision'][i]:>10.3f}{r['recall'][i]:>10.3f}{r['f1'][i]:>8.3f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(r["cm"])


def area_table(actual, pred, class_names, px_km2=0.0009):
    """Area per class in km2 (one 30 m pixel = 900 m2 = 0.0009 km2)."""
    n = len(class_names)
    a = np.bincount(actual.astype(int), minlength=n)[:n] * px_km2
    p = np.bincount(pred.astype(int), minlength=n)[:n] * px_km2
    print(f"{'class':<18}{'actual km2':>12}{'predicted km2':>15}{'difference %':>14}")
    for i, c in enumerate(class_names):
        diff = 100 * (p[i] - a[i]) / max(a[i], 1e-9)
        print(f"{c:<18}{a[i]:>12.1f}{p[i]:>15.1f}{diff:>14.1f}")