# train.py - train ONE model, test it, and write the predicted 2023 GeoTIFF
# usage:  python src/train.py unet
#         python src/train.py pixel_lstm
#         python src/train.py convlstm_unet
# options: --epochs 15  --lc_weight 0.8  --per_epoch 600  --batch 8  --patch 64
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from config import get_area, LC_WEIGHT
from splits import block_split
from models import Forecaster
from dl_utils import PatchDataset, candidate_positions, predict_map
from metrics import evaluate, print_report, area_table
from export_tif import save_prediction_tif
from plots import save_comparison_figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["unet", "pixel_lstm", "convlstm_unet"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lc_weight", type=float, default=LC_WEIGHT)   # aux weight = 1 - this
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--per_epoch", type=int, default=600)           # patches per epoch
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = get_area()
    out_dir = cfg["output_dir"]
    ds = np.load(out_dir / "dataset.npz")
    lulc, aux, valid = ds["lulc"], ds["aux"], ds["valid"]
    names = [cfg["classes"][c] for c in sorted(cfg["classes"])]
    n = len(names)

    maps_in = lulc[:-1]                    # 1993, 2003, 2013
    target = lulc[-1].astype(np.int64)     # real 2023
    prev = lulc[-2].astype(np.int64)       # 2013
    train, val, test = block_split(valid)
    P = args.patch

    train_pos = candidate_positions(train, P, 16, 0.3)
    val_pos = candidate_positions(val, P, 32, 0.3)
    print(f"patches: train {len(train_pos)} | val {len(val_pos)}")
    train_ds = PatchDataset(maps_in, aux, target, train, train_pos, P, n, augment=True)
    val_ds = PatchDataset(maps_in, aux, target, val, val_pos, P, n, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    k = min(len(train_ds), args.per_epoch)

    model = Forecaster(args.model, n, maps_in.shape[0], aux.shape[0], lc_weight=args.lc_weight)
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"{args.model}_w{int(round(args.lc_weight * 100))}"
    if args.seed != 0:
        tag += f"_s{args.seed}"
    print(f"model {tag}: {n_params:,} parameters | land-cover weight {args.lc_weight} "
          f"| aux weight {1 - args.lc_weight:.2f}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    best_path, best_val = model_dir / f"{tag}.pt", 1e9
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- training ----
        model.train()
        loader = DataLoader(train_ds, batch_size=args.batch,
                            sampler=RandomSampler(train_ds, num_samples=k))
        tl, nb = 0.0, 0
        for seq, a, y in loader:
            opt.zero_grad()
            loss = lossf(model(seq, a), y)
            loss.backward()
            opt.step()
            tl, nb = tl + loss.item(), nb + 1

        # ---- validation ----
        model.eval()
        vl, vb, correct, total = 0.0, 0, 0, 0
        with torch.no_grad():
            for seq, a, y in val_loader:
                logits = model(seq, a)
                vl, vb = vl + lossf(logits, y).item(), vb + 1
                m = y != -100
                correct += (logits.argmax(1)[m] == y[m]).sum().item()
                total += m.sum().item()
        vl /= vb
        flag = ""
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_path)
            flag = "  <- best so far (saved)"
        print(f"epoch {epoch:2d}/{args.epochs} | train loss {tl / nb:.4f} | val loss {vl:.4f} "
              f"| val acc {100 * correct / total:.1f}% | {(time.time() - t0) / 60:.1f} min{flag}")

    minutes = (time.time() - t0) / 60

    # ---- final test with the best epoch ----
    model.load_state_dict(torch.load(best_path))
    print("\nPredicting the whole 2023 map (tile by tile) ...")
    pred = predict_map(model, maps_in, aux, n)

    r = evaluate(target[test], pred[test], prev[test], n)
    print_report(f"{tag}, TEST blocks (never seen in training)", r, names)
    print("\nArea per class on the TEST blocks (km2):")
    area_table(target[test], pred[test], names)

    np.save(out_dir / f"pred_2023_{tag}.npy", pred)
    save_prediction_tif(pred, ds, cfg, out_dir / f"pred_2023_{tag}.tif")
    save_comparison_figure(target, pred, valid, test, cfg, out_dir / f"{tag}_vs_actual_2023.png",
                           f"{tag}: predicted 2023 vs actual 2023")
    with open(out_dir / f"info_{tag}.json", "w") as f:
        json.dump({"params": n_params, "minutes": round(minutes, 1)}, f)


if __name__ == "__main__":
    main()