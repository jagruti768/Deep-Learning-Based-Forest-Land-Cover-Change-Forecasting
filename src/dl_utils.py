# dl_utils.py - data loading (patches) and full-map prediction for the deep models
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import uniform_filter


def to_onehot(maps, n):
    """(..., H, W) integer maps -> (..., n, H, W) float. Values >= n (nodata) -> all zeros."""
    maps = torch.where(maps >= n, torch.full_like(maps, n), maps)
    oh = F.one_hot(maps, n + 1)[..., :n]
    return oh.movedim(-1, -3).float()


def candidate_positions(mask, patch, stride, min_frac):
    """Top-left corners of patches that contain at least min_frac pixels of `mask`."""
    H, W = mask.shape
    frac = uniform_filter(mask.astype("float32"), size=patch, mode="constant")
    pos = []
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            if frac[y + patch // 2, x + patch // 2] >= min_frac:
                pos.append((y, x))
    return pos


class PatchDataset(Dataset):
    """Cuts small square patches out of the big map."""
    def __init__(self, maps, aux, target, mask, positions, patch, n, augment):
        self.maps, self.aux, self.target, self.mask = maps, aux, target, mask
        self.positions, self.P, self.n, self.augment = positions, patch, n, augment

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        y, x = self.positions[i]
        P = self.P
        m = self.maps[:, y:y + P, x:x + P]
        a = self.aux[:, y:y + P, x:x + P]
        # the loss only counts pixels that belong to this split (others = -100 = ignored)
        t = np.where(self.mask[y:y + P, x:x + P], self.target[y:y + P, x:x + P], -100)
        if self.augment:                              # random rotation / flip
            k = np.random.randint(4)
            m, a, t = (np.rot90(m, k, axes=(1, 2)), np.rot90(a, k, axes=(1, 2)), np.rot90(t, k))
            if np.random.rand() < 0.5:
                m, a, t = m[:, :, ::-1], a[:, :, ::-1], t[:, ::-1]
        m = torch.from_numpy(np.ascontiguousarray(m).astype(np.int64))
        a = torch.from_numpy(np.ascontiguousarray(a).astype(np.float32))
        t = torch.from_numpy(np.ascontiguousarray(t).astype(np.int64))
        return to_onehot(m, self.n), a, t


@torch.no_grad()
def predict_map(model, maps_in, aux, n, tile=256, margin=32):
    """Predict the whole map tile by tile (a big map does not fit in memory at once).
    maps_in (T,H,W) uint8, aux (C,H,W) float32 -> (H,W) uint8 predicted classes."""
    model.eval()
    T, H, W = maps_in.shape
    Hp, Wp = int(np.ceil(H / tile)) * tile, int(np.ceil(W / tile)) * tile
    m_pad = np.full((T, Hp + 2 * margin, Wp + 2 * margin), 255, dtype=np.uint8)
    a_pad = np.zeros((aux.shape[0], Hp + 2 * margin, Wp + 2 * margin), dtype=np.float32)
    m_pad[:, margin:margin + H, margin:margin + W] = maps_in
    a_pad[:, margin:margin + H, margin:margin + W] = aux

    out = np.zeros((Hp, Wp), dtype=np.uint8)
    size = tile + 2 * margin                           # 320, divisible by 4
    for y in range(0, Hp, tile):
        for x in range(0, Wp, tile):
            m = torch.from_numpy(m_pad[:, y:y + size, x:x + size].astype(np.int64))
            a = torch.from_numpy(a_pad[:, y:y + size, x:x + size])
            logits = model(to_onehot(m, n)[None], a[None])
            pred = logits.argmax(1)[0, margin:margin + tile, margin:margin + tile]
            out[y:y + tile, x:x + tile] = pred.numpy().astype(np.uint8)
    return out[:H, :W]