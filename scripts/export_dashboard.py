"""Export the small files the dashboard reads into reports/dashboard/.

    python scripts/export_dashboard.py

The hosted app has no price data, panel or models, only these files:

    returns_<period>.parquet   per strategy and day: gross return and turnover, with and without a
                               one-day execution delay, net exposure, and the market return. Costs are
                               linear, so the app computes net = gross - cost * turnover for any cost.
    signals_<period>.parquet   per stock and day: OU s-score, OU position, and each model's score
                               (standardized across stocks each day, averaged over the 3 seeds)
    trials.csv                 every Fold 1 configuration tried, with its validation Sharpe
"""

import json

import numpy as np
import pandas as pd

from sharpee.backtest.engine import run_backtest
from sharpee.config import load_config, repo_path
from sharpee.evaluation.strategies import combined, strategy_scores
from sharpee.pipeline import load_panel, runs_dir
from sharpee.strategies.ou_strategy import ou_fit
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


def main():
    cfg = load_config("data", "experiment", "baseline")
    runs = runs_dir(cfg)
    out = repo_path(cfg["reports_dir"]) / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    panel = load_panel(cfg)
    export_period(panel, runs, cfg, "test", cfg["test_years"], out)
    export_period(panel, runs, cfg, "holdout", cfg["holdout_years"], out)

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


if __name__ == "__main__":
    main()
