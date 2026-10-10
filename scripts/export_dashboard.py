"""Export the small files the dashboard reads into reports/dashboard/.

    python scripts/export_dashboard.py                  # everything
    python scripts/export_dashboard.py --only attention # one part

The hosted app has no price data, panel or models, only these files:

    returns_<period>.parquet   per strategy and day: gross return and turnover, with and without a
                               one-day execution delay, net exposure, and the market return. Costs are
                               linear, so the app computes net = gross - cost * turnover for any cost.
    signals_<period>.parquet   per stock and day: OU s-score, OU position, and each model's score
                               (standardized across stocks each day, averaged over the 3 seeds)
    trials.csv                 every Fold 1 configuration tried, with its validation Sharpe
    attention_test.csv         Transformer attention received by each day of its window (seed 0, test
                               years), for moderate and extreme OU stretches
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from sharpee.backtest.engine import run_backtest
from sharpee.config import load_config, repo_path
from sharpee.evaluation.strategies import combined, strategy_scores
from sharpee.evaluation.walk_forward import make_folds, split
from sharpee.models import build_model
from sharpee.pipeline import load_panel, runs_dir
from sharpee.strategies.ou_strategy import ou_fit
from sharpee.training.dataset import DayBlocks
from sharpee.training.trials import load_trials

MODELS = ["mlp", "temporal_cnn", "transformer"]


def standardized(scores: np.ndarray, tradable: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score per day over tradable stocks; NaN elsewhere."""
    s = np.where(tradable, scores.astype(np.float64), np.nan)
    return (s - np.nanmean(s, axis=1, keepdims=True)) / np.nanstd(s, axis=1, keepdims=True)


def export_period(panel, runs, cfg, period, years, out):
    idx = panel.index_of(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    scores = strategy_scores(panel, runs, period, idx)

    cols = {}
    for name, seeds in scores.items():
        for delay in (0, 1):
            book = combined([run_backtest(panel, idx, s, cost_bps=0, delay=delay, cap=cfg["cap"],
                                          center=name != "ou") for _, s in seeds])
            cols[(name, f"gross_d{delay}")] = book["gross"]
            cols[(name, f"turnover_d{delay}")] = book["turnover"]
            if delay == 0:
                cols[(name, "net_exposure")] = book["net_exposure"]
                cols[("market", "return")] = book["market"]
    returns = pd.DataFrame(cols)
    returns.columns = [f"{a}.{b}" for a, b in returns.columns]
    returns.astype("float32").to_parquet(out / f"returns_{period}.parquet")

    tradable = panel.tradable[idx]
    s_score, _, valid = ou_fit(panel.windows(idx, cfg["window"]))
    frame = {"s_score": np.where(valid, s_score, np.nan), "ou": scores["ou"][0][1]}
    for name in MODELS:
        if name in scores:
            frame[name] = np.nanmean([standardized(s, tradable) for _, s in scores[name]], axis=0)
    uni = panel.universe[panel.month[idx]]
    keep = tradable & (uni >= 0)
    rows, slots = np.nonzero(keep)
    signals = pd.DataFrame({
        "date": panel.dates[idx][rows],
        "ticker": pd.Categorical(panel.tickers[uni[rows, slots]]),
        **{k: v[rows, slots].astype("float32") for k, v in frame.items()},
    })
    signals.to_parquet(out / f"signals_{period}.parquet", index=False)
    print(f"{period}: {len(returns)} days, {returns.shape[1]} return columns; "
          f"{len(signals):,} stock-days across {signals['ticker'].nunique()} tickers")


@torch.no_grad()
def export_attention(panel, runs, cfg, out):
    """Last-layer attention received per window day, averaged over heads, query days and stock-days."""
    best = json.loads((runs / "best_transformer.json").read_text())["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parts = {"moderate": [], "extreme": []}
    for fold in make_folds(cfg["test_years"], cfg["train_start"], cfg["val_years"]):
        _, _, te = split(panel, fold, cfg["embargo_days"])
        model = build_model("transformer", best["lookback"], **best["params"]).to(device).eval()
        model.load_state_dict(torch.load(runs / "walkforward" / "test" / "models" /
                                         f"transformer_seed0_{fold.name}.pt", map_location=device))
        x = DayBlocks(panel, te, best["lookback"], device).X
        s, _, ok = ou_fit(panel.windows(te, cfg["window"]))
        ok &= panel.tradable[te]
        flat_x, extreme = x[torch.from_numpy(ok).to(device)], np.abs(s[ok]) >= 2
        for a in range(0, len(flat_x), 4096):
            _, attn = model(flat_x[a:a + 4096].unsqueeze(-1), return_attn=True)
            got = attn[-1].mean(dim=(1, 2)).cpu().numpy()
            parts["moderate"].append(got[~extreme[a:a + 4096]])
            parts["extreme"].append(got[extreme[a:a + 4096]])
    days = np.arange(best["lookback"]) - best["lookback"] + 1
    pd.DataFrame({"day": days, **{k: np.concatenate(v).mean(0) for k, v in parts.items()}})         .to_csv(out / "attention_test.csv", index=False)
    print("attention: written")


def export_trials(runs, out):
    trials = load_trials(runs)
    trials = trials[trials["fold"] == "test_2020"].copy()

    def describe(row):
        p = json.loads(row.params)
        if row.model == "ou":
            return f"entry {p['entry']}"
        size = {k: v for k, v in p["params"].items() if k in ("hidden", "channels", "d_model")}
        return ", ".join(f"{k} {v}" for k, v in size.items()) + f", lr {p['train']['lr']}"

    trials["config"] = trials.apply(describe, axis=1)
    trials[["model", "config", "val_sharpe", "trial_id"]].to_csv(out / "trials.csv", index=False)
    print(f"trials: {len(trials)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["periods", "trials", "attention"])
    args = ap.parse_args()
    cfg = load_config("data", "experiment", "baseline")
    runs = runs_dir(cfg)
    out = repo_path(cfg["reports_dir"]) / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    panel = load_panel(cfg)
    if args.only in (None, "periods"):
        export_period(panel, runs, cfg, "test", cfg["test_years"], out)
        export_period(panel, runs, cfg, "holdout", cfg["holdout_years"], out)
    if args.only in (None, "trials"):
        export_trials(runs, out)
    if args.only in (None, "attention"):
        export_attention(panel, runs, cfg, out)


if __name__ == "__main__":
    main()
