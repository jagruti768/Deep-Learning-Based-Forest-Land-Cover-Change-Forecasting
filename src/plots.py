# plots.py - actual vs predicted vs error picture
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def save_comparison_figure(actual, pred, valid, test_mask, cfg, path, title):
    classes = sorted(cfg["classes"])
    cmap = ListedColormap([cfg["colors"][c] for c in classes])
    fig, ax = plt.subplots(1, 3, figsize=(18, 8))

    for a, m, t in ((ax[0], actual, "Actual"), (ax[1], pred, "Predicted")):
        a.imshow(np.ma.masked_where(~valid, m), cmap=cmap, vmin=0,
                 vmax=len(classes) - 1, interpolation="nearest")
        a.set_title(t)

    wrong = np.ma.masked_where(~valid, (pred != actual).astype(float))
    ax[2].imshow(wrong, cmap=ListedColormap(["#dddddd", "#d62728"]),
                 vmin=0, vmax=1, interpolation="nearest")
    ax[2].contour(test_mask.astype(float), levels=[0.5], colors="blue", linewidths=0.4)
    ax[2].set_title("Error (red = wrong, blue lines = held-out TEST blocks)")

    for a in ax:
        a.axis("off")
    fig.legend(handles=[Patch(color=cfg["colors"][c], label=cfg["classes"][c]) for c in classes],
               loc="lower center", ncol=len(classes))
    fig.suptitle(title)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")