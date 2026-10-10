"""Walk-forward retraining with the frozen Fold 1 settings (docs/phase3_protocol.md).

    python scripts/walk_forward.py --model transformer               # test years 2020-2023, seeds 0 1 2
    python scripts/walk_forward.py --model transformer --no-cost     # ablation: no costs in the loss
    python scripts/walk_forward.py --model transformer --holdout     # 2024-2025, run once at the very end

For every seed and test year, a model is trained from scratch on the fold's
training years, early-stopped on its validation years, and scores the test
year. Scores are saved per seed in runs/walkforward/<period>/; runs that
already exist are skipped, so an interrupted job can simply be restarted.
"""

import argparse
import json
import time

import numpy as np
import torch

from sharpee.config import load_config
from sharpee.evaluation.walk_forward import make_fold, make_folds, split
from sharpee.models import MODELS
from sharpee.pipeline import load_panel, runs_dir
from sharpee.training.dataset import DayBlocks
from sharpee.training.train import predict
from sharpee.training.tuning import train_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-cost", action="store_true", help="ablation: train with train_cost_bps = 0")
    ap.add_argument("--holdout", action="store_true", help="the one-shot 2024-2025 run")
    args = ap.parse_args()

    cfg = load_config("data", "experiment")
    runs = runs_dir(cfg)
    model_cfg = json.loads((runs / f"best_{args.model}.json").read_text())["config"]
    name = args.model
    if args.no_cost:
        model_cfg["train"]["train_cost_bps"] = 0.0
        name += "_nocost"

    if args.holdout:
        years = cfg["holdout_years"]
        folds, period = [make_fold(years[0], years[-1], cfg["train_start"], cfg["val_years"])], "holdout"
    else:
        folds, period = make_folds(cfg["test_years"], cfg["train_start"], cfg["val_years"]), "test"
    out = runs / "walkforward" / period
    (out / "models").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = load_panel(cfg)
    lookback = model_cfg["lookback"]

    for seed in args.seeds:
        target = out / f"{name}_seed{seed}.npz"
        if target.exists():
            print(f"{name} seed {seed}: already done, skipping")
            continue
        idx_all, scores_all, folds_log = [], [], []
        for fold in folds:
            tr, va, te = split(panel, fold, cfg["embargo_days"])
            start = time.time()
            model, res, _ = train_config(model_cfg, DayBlocks(panel, tr, lookback, device),
                                         DayBlocks(panel, va, lookback, device), cfg["cap"],
                                         cfg["cost_bps"], seed, device, log=lambda _: None)
            idx_all.append(te)
            scores_all.append(predict(model, DayBlocks(panel, te, lookback, device)))
            torch.save(model.state_dict(), out / "models" / f"{name}_seed{seed}_{fold.name}.pt")
            folds_log.append({"fold": fold.name, "train": fold.train, "val": fold.val, "test": fold.test,
                              "best_epoch": res["best_epoch"], "val_sharpe": res["best_val_sharpe"],
                              "seconds": round(time.time() - start)})
            print(f"{name} seed {seed} {fold.name}: val net Sharpe {res['best_val_sharpe']:+.2f} "
                  f"at epoch {res['best_epoch']}, {time.time() - start:.0f}s", flush=True)
        idx = np.concatenate(idx_all)
        np.savez_compressed(target, idx=idx, dates=panel.dates[idx].values, scores=np.concatenate(scores_all))
        (out / f"{name}_seed{seed}.json").write_text(json.dumps(
            {"model": args.model, "name": name, "seed": seed, "config": model_cfg, "folds": folds_log}, indent=2))


if __name__ == "__main__":
    main()
