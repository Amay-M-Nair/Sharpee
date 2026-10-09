"""Run the OU baseline over every date and report it on the walk-forward test years.

    python scripts/run_baseline.py               # test years 2020-2023
    python scripts/run_baseline.py --holdout     # 2024-2025: run once, after every setting is frozen

The OU strategy has no fitted parameters (Avellaneda-Lee defaults), so its
positions are computed once over the whole panel and then sliced by period.
"""

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sharpee.backtest.engine import run_backtest  # noqa: E402
from sharpee.config import load_config, repo_path  # noqa: E402
from sharpee.evaluation.metrics import summarize  # noqa: E402
from sharpee.features.diagnostics import residual_report  # noqa: E402
from sharpee.pipeline import load_panel, runs_dir  # noqa: E402
from sharpee.strategies.ou_strategy import ou_positions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--rebuild-panel", action="store_true")
    args = ap.parse_args()
    cfg = load_config("data", "experiment", "baseline")
    reports = repo_path(cfg["reports_dir"])
    (reports / "figures").mkdir(parents=True, exist_ok=True)

    panel = load_panel(cfg, rebuild=args.rebuild_panel)
    print(f"panel: {len(panel.dates)} dates {panel.dates[0].date()} -> {panel.dates[-1].date()}, "
          f"{panel.n_slots} slots, {len(panel.phi)} monthly factor models")

    all_idx = np.arange(len(panel.dates))
    pos = ou_positions(panel, all_idx, window=cfg["window"], entry=cfg["entry"],
                       exit_short=cfg["exit_short"], exit_long=cfg["exit_long"],
                       kappa_min=cfg["kappa_min"])
    np.savez_compressed(runs_dir(cfg) / "scores_ou.npz", dates=panel.dates.values, scores=pos)

    years = cfg["holdout_years"] if args.holdout else cfg["test_years"]
    period = panel.index_of(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    tag = "holdout" if args.holdout else "test"
    print(f"\nresidual diagnostics, {years[0]}-{years[-1]}")
    print(residual_report(panel, period, cfg["window"], cfg["kappa_min"]).round(3).to_string())

    rows, frames = [], {}
    for cost in cfg["cost_grid"]:
        for delay in (0, 1):
            f = run_backtest(panel, period, pos[period], cost_bps=cost, delay=delay, cap=cfg["cap"])
            rows.append({"strategy": "ou", "period": f"{years[0]}-{years[-1]}", "cost_bps": cost,
                         "delay": delay, **summarize(f)})
            frames[(cost, delay)] = f
    for y in years:
        idx = panel.index_of(f"{y}-01-01", f"{y}-12-31")
        f = run_backtest(panel, idx, pos[idx], cost_bps=cfg["cost_bps"], cap=cfg["cap"])
        rows.append({"strategy": "ou", "period": str(y), "cost_bps": cfg["cost_bps"], "delay": 0,
                     **summarize(f)})

    table = pd.DataFrame(rows)
    table.to_csv(reports / f"baseline_ou_{tag}.csv", index=False)
    main_frame = frames[(cfg["cost_bps"], 0)]
    main_frame.to_parquet(runs_dir(cfg) / f"daily_ou_{tag}.parquet")
    cols = ["period", "cost_bps", "delay", "ann_return", "ann_vol", "sharpe", "gross_sharpe",
            "max_drawdown", "avg_turnover", "beta", "corr_market", "net_exposure_mean"]
    print(f"\nOU baseline, {tag} period")
    print(table[cols].round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4))
    for cost in cfg["cost_grid"]:
        eq = (1 + frames[(cost, 0)]["net"]).cumprod()
        ax.plot(eq.index, eq.values, label=f"net, {cost} bps")
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_title(f"OU baseline equity, {years[0]}-{years[-1]}")
    ax.set_ylabel("growth of 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "figures" / f"ou_equity_{tag}.png", dpi=120)
    print(f"\nwrote {reports / f'baseline_ou_{tag}.csv'} and figures/ou_equity_{tag}.png")


if __name__ == "__main__":
    main()
