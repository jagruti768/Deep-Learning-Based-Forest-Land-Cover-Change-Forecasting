# splits.py - split the map into spatial BLOCKS (not random pixels)
import numpy as np
from config import AREAS, STUDY_AREA


def block_split(valid, block=None, test_frac=0.3, val_frac=0.1, seed=42):
    """Return three boolean maps (train, val, test) the same shape as the map.
    The block size (pixels) comes from config.py ("block") unless it is given."""
    if block is None:
        block = AREAS[STUDY_AREA].get("block", 128)
    H, W = valid.shape
    n_cols = int(np.ceil(W / block))
    n_rows = int(np.ceil(H / block))
    # every pixel gets the number of the block it sits in
    block_id = (np.arange(H) // block)[:, None] * n_cols + (np.arange(W) // block)[None, :]

    rng = np.random.default_rng(seed)           # fixed seed -> same split every run
    order = rng.permutation(n_rows * n_cols)
    n_test = int(round(test_frac * n_rows * n_cols))
    n_val = int(round(val_frac * n_rows * n_cols))

    role = np.zeros(n_rows * n_cols, dtype=np.int8)   # 0=train, 1=val, 2=test
    role[order[:n_test]] = 2
    role[order[n_test:n_test + n_val]] = 1
    pixel_role = role[block_id]

    return valid & (pixel_role == 0), valid & (pixel_role == 1), valid & (pixel_role == 2)