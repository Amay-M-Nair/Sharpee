"""Ridge and gradient-boosting baselines (docs/phase3_amendment_baselines.md).

    python scripts/train_tabular.py --model gbm --tune                    # Fold 1 grid, every config logged
    python scripts/train_tabular.py --model gbm --walk-forward            # 2020-2023 with the frozen best config
    python scripts/train_tabular.py --model gbm --walk-forward --holdout  # 2024-2025, run once

Same inputs, folds, embargoes, portfolio layer and costs as the networks. The
models are deterministic, so each fold is fitted once (saved as seed 0).
"""

import argparse
import json
import time

import numpy as np

from sharpee.backtest.engine import run_backtest
from sharpee.config import load_config
from sharpee.evaluation.metrics import sharpe
from sharpee.evaluation.walk_forward import make_fold, make_folds, split
from sharpee.pipeline import load_panel, runs_dir
from sharpee.training.tabular import TABULAR, fit, make_xy, predict_scores
from sharpee.training.trials import log_trial
from sharpee.training.tuning import expand


def fit_fold(panel, cfg, model_cfg, fold):
    tr, va, te = split(panel, fold, cfg["embargo_days"])
    x, y = make_xy(panel, tr, model_cfg["lookback"])
    model = fit(model_cfg["model"], model_cfg["params"], x, y)
    return model, va, te


def tune(panel, cfg, model_cfg, runs):
    fold = make_folds(cfg["test_years"], cfg["train_start"], cfg["val_years"])[0]
    results = []
    for k, c in enumerate(expand(model_cfg), 1):
        start = time.time()
        model, va, _ = fit_fold(panel, cfg, c, fold)
        frame = run_backtest(panel, va, predict_scores(model, panel, va, c["lookback"]),
                             cost_bps=cfg["cost_bps"], cap=cfg["cap"])
        trial_id = log_trial(runs, c["model"], fold.name, {"params": c["params"]}, frame["net"])
        s = sharpe(frame["net"])
        print(f"[{k}] {c['model']} {c['params']}: val net Sharpe {s:+.2f}, gross {sharpe(frame['gross']):+.2f}, "
              f"turnover {frame['turnover'].mean():.2f}, trial {trial_id}, {time.time() - start:.0f}s", flush=True)
        results.append((s, trial_id, c))
    best_sharpe, best_id, best_cfg = max(results, key=lambda r: r[0])
    (runs / f"best_{model_cfg['model']}.json").write_text(json.dumps(
        {"trial_id": best_id, "val_sharpe": best_sharpe, "config": best_cfg}, indent=2))
    print(f"best {model_cfg['model']}: {best_cfg['params']} val net Sharpe {best_sharpe:+.2f}")


def walk_forward(panel, cfg, name, runs, holdout):
    model_cfg = json.loads((runs / f"best_{name}.json").read_text())["config"]
    if holdout:
        years = cfg["holdout_years"]
        folds, period = [make_fold(years[0], years[-1], cfg["train_start"], cfg["val_years"])], "holdout"
    else:
        folds, period = make_folds(cfg["test_years"], cfg["train_start"], cfg["val_years"]), "test"
    out = runs / "walkforward" / period
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}_seed0.npz"
    if target.exists():
        print(f"{name}: already done, skipping")
        return
    idx_all, scores_all, log = [], [], []
    for fold in folds:
        start = time.time()
        model, va, te = fit_fold(panel, cfg, model_cfg, fold)
        val = run_backtest(panel, va, predict_scores(model, panel, va, model_cfg["lookback"]),
                           cost_bps=cfg["cost_bps"], cap=cfg["cap"])
        idx_all.append(te)
        scores_all.append(predict_scores(model, panel, te, model_cfg["lookback"]))
        log.append({"fold": fold.name, "train": fold.train, "val": fold.val, "test": fold.test,
                    "val_sharpe": sharpe(val["net"]), "seconds": round(time.time() - start)})
        print(f"{name} {fold.name}: val net Sharpe {log[-1]['val_sharpe']:+.2f}, {log[-1]['seconds']}s", flush=True)
    idx = np.concatenate(idx_all)
    np.savez_compressed(target, idx=idx, dates=panel.dates[idx].values, scores=np.concatenate(scores_all))
    (out / f"{name}_seed0.json").write_text(json.dumps(
        {"model": name, "name": name, "seed": 0, "config": model_cfg, "folds": log}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=TABULAR)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    cfg = load_config("data", "experiment")
    runs = runs_dir(cfg)
    panel = load_panel(cfg)
    if args.tune:
        tune(panel, cfg, load_config(args.model), runs)
    if args.walk_forward:
        walk_forward(panel, cfg, args.model, runs, args.holdout)


if __name__ == "__main__":
    main()
