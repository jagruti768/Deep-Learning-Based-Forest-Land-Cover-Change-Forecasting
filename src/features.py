# features.py - turn each pixel into a row of numbers for the models
import numpy as np
from scipy.ndimage import uniform_filter


def build_features(maps, aux, n_classes, window=7):
    """
    maps : (T, H, W) class maps in time order, e.g. 1993, 2003, 2013
    aux  : (C, H, W) auxiliary layers scaled 0..1 (slope, later elevation)
    returns (H, W, F) float32 array
    """
    classes = np.arange(n_classes)
    parts = []

    # 1) class of the pixel in each past year, one-hot (e.g. Urban = 0,1,0,0,0)
    for m in maps:
        parts.append((m[..., None] == classes).astype("float32"))

    # 2) what surrounds the pixel in the most recent year:
    #    fraction of each class inside a window x window neighbourhood
    last = maps[-1]
    for c in classes:
        frac = uniform_filter((last == c).astype("float32"), size=window)
        parts.append(frac[..., None])

    # 3) auxiliary layers (slope)
    parts.append(np.moveaxis(aux, 0, -1).astype("float32"))

    return np.concatenate(parts, axis=-1)