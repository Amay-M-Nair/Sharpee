"""Tune a neural model on walk-forward Fold 1; every configuration is logged as a trial.

    python scripts/train_model.py --model transformer --tune   # full grid from configs/transformer.yaml
    python scripts/train_model.py --model mlp                  # base config only
    python scripts/train_model.py --model transformer --tune --resume   # skip configs already logged

Only Fold 1's training (2010-2017) and validation (2018-2019) years are used.
The test years stay untouched until Phase 3.
"""

import argparse
import json
import time

import torch

from sharpee.config import load_config
from sharpee.evaluation.walk_forward import make_folds, split
from sharpee.models import MODELS
from sharpee.pipeline import load_panel, runs_dir
from sharpee.training.dataset import DayBlocks
from sharpee.training.trials import load_trials, log_trial
from sharpee.training.tuning import expand, train_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="print every epoch")
    ap.add_argument("--resume", action="store_true",
                    help="skip configurations already in the trial log (e.g. after an interrupted run), "
                         "so none is logged twice")
    args = ap.parse_args()

    cfg = load_config("data", "experiment")
    model_cfg = load_config(args.model)
    if not args.tune:
        model_cfg.pop("tune", None)
    seed = cfg["seed"] if args.seed is None else args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    panel = load_panel(cfg)
    fold = make_folds(cfg["test_years"], cfg["train_start"], cfg["val_years"])[0]
    tr, va, _ = split(panel, fold, cfg["embargo_days"])
    lookback = model_cfg["lookback"]
    train, val = DayBlocks(panel, tr, lookback, device), DayBlocks(panel, va, lookback, device)
    print(f"{fold.name}: train {panel.dates[tr[0]].date()}..{panel.dates[tr[-1]].date()} ({len(tr)} days), "
          f"validate {panel.dates[va[0]].date()}..{panel.dates[va[-1]].date()} ({len(va)} days), {device}")

    runs = runs_dir(cfg)
    logged = {}
    if args.resume:
        t = load_trials(runs)
        t = t[(t["model"] == args.model) & (t["fold"] == fold.name)]
        logged = {row.params: (row.val_sharpe, row.trial_id) for row in t.itertuples()}
    results = []
    for k, c in enumerate(expand(model_cfg), 1):
        print(f"[{k}] {args.model} params={c['params']} lr={c['train']['lr']}")
        key = json.dumps({"params": c["params"], "train": c["train"], "seed": seed}, sort_keys=True)
        if key in logged:
            sharpe, trial_id = logged[key]
            print(f"    already logged: val net Sharpe {sharpe:+.2f}, trial {trial_id}")
            results.append((sharpe, trial_id, c, None))
            continue
        start = time.time()
        model, res, val_net = train_config(c, train, val, cfg["cap"], cfg["cost_bps"], seed, device,
                                           log=print if args.verbose else (lambda _: None))
        trial_id = log_trial(runs, args.model, fold.name,
                             {"params": c["params"], "train": c["train"], "seed": seed}, val_net)
        (runs / "trials" / f"{trial_id}_history.json").write_text(json.dumps(res["history"]))
        print(f"    val net Sharpe {res['best_val_sharpe']:+.2f} at epoch {res['best_epoch']}, "
              f"trial {trial_id}, {time.time() - start:.0f}s")
        results.append((res["best_val_sharpe"], trial_id, c, model))

    best_sharpe, best_id, best_cfg, best_model = max(results, key=lambda r: r[0])
    (runs / f"best_{args.model}.json").write_text(json.dumps(
        {"trial_id": best_id, "val_sharpe": best_sharpe, "seed": seed, "config": best_cfg}, indent=2))
    if best_model is not None:  # None when the best config was trained in an earlier, interrupted run
        torch.save(best_model.state_dict(), runs / f"{args.model}_fold1.pt")
    print(f"best {args.model}: val net Sharpe {best_sharpe:+.2f} (trial {best_id}) -> runs/best_{args.model}.json")


if __name__ == "__main__":
    main()
