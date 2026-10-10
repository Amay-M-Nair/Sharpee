"""Phase 3 evaluation: backtest every strategy on the test years and apply the pre-registered rules.

    python scripts/run_backtest.py              # 2020-2023, from runs/walkforward/test
    python scripts/run_backtest.py --holdout    # 2024-2025, run once at the very end

Rules, metrics and seed handling follow docs/phase3_protocol.md: each neural
model's seeds are combined with one third of capital in each seed's book, and
the Transformer is the primary candidate. Writes reports/phase3/<period>_*.
"""

import argparse

import numpy as np
import pandas as pd

from sharpee.backtest.engine import run_backtest
from sharpee.config import load_config, repo_path
from sharpee.evaluation.metrics import summarize
from sharpee.evaluation.significance import (daily_sharpe, dsr, paired_sharpe_difference, psr,
                                             stationary_bootstrap_ci)
from sharpee.evaluation.strategies import combined, strategy_scores
from sharpee.pipeline import load_panel, runs_dir
from sharpee.training.trials import load_trials

PRIMARY = "transformer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    cfg = load_config("data", "experiment")
    runs = runs_dir(cfg)
    period = "holdout" if args.holdout else "test"
    years = cfg["holdout_years"] if args.holdout else cfg["test_years"]
    out = repo_path(cfg["reports_dir"]) / "phase3"
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(cfg)
    idx = panel.index_of(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    scores = strategy_scores(panel, runs, period, idx)
    print(f"{period} {panel.dates[idx[0]].date()}..{panel.dates[idx[-1]].date()} ({len(idx)} days): "
          f"{', '.join(scores)}")

    trials = load_trials(runs)
    trials = trials[trials["fold"] == "test_2020"]
    n_trials, var_sr = len(trials), float(trials["val_sr_daily"].var())

    summary, per_seed, per_year, daily = [], [], [], {}
    for name, seeds in scores.items():
        center = name != "ou"
        for cost in cfg["cost_grid"]:
            for delay in (0, 1):
                frames = [run_backtest(panel, idx, s, cost_bps=cost, delay=delay, cap=cfg["cap"], center=center)
                          for _, s in seeds]
                book = combined(frames)
                summary.append({"strategy": name, "cost_bps": cost, "delay": delay, **summarize(book)})
                if cost == cfg["cost_bps"] and delay == 0:
                    for (seed, _), f in zip(seeds, frames):
                        per_seed.append({"strategy": name, "seed": seed, **summarize(f)})
                    for year, g in book.groupby(book.index.year):
                        per_year.append({"strategy": name, "year": year, **summarize(g)})
                    daily[name] = book

    base = daily["ou"]["net"].to_numpy()
    sig = []
    for name, book in daily.items():
        r = book["net"].to_numpy()
        lo, hi = stationary_bootstrap_ci(r)
        row = {"strategy": name, "net_sharpe": daily_sharpe(r) * np.sqrt(252), "ci_low": lo, "ci_high": hi,
               "psr": psr(r), "dsr": dsr(r, n_trials, var_sr), "n_trials": n_trials}
        if name != "ou":
            d, d_lo, d_hi = paired_sharpe_difference(r, base)
            row.update(diff_vs_ou=d, diff_ci_low=d_lo, diff_ci_high=d_hi)
        sig.append(row)
    sig = pd.DataFrame(sig)

    pd.DataFrame(summary).to_csv(out / f"{period}_summary.csv", index=False)
    pd.DataFrame(per_seed).to_csv(out / f"{period}_per_seed.csv", index=False)
    pd.DataFrame(per_year).to_csv(out / f"{period}_per_year.csv", index=False)
    sig.to_csv(out / f"{period}_significance.csv", index=False)
    pd.concat({k: v for k, v in daily.items()}, axis=1).to_parquet(out / f"{period}_daily.parquet")

    view = pd.DataFrame(summary)
    view = view[(view.cost_bps == cfg["cost_bps"]) & (view.delay == 0)].set_index("strategy")
    print("\nnet of 5 bps, seeds combined")
    print(view[["ann_return", "ann_vol", "sharpe", "gross_sharpe", "max_drawdown", "avg_turnover", "beta"]]
          .round(3).to_string())
    print("\nsignificance")
    print(sig.set_index("strategy").round(3).to_string())

    if PRIMARY in daily:
        p = sig.set_index("strategy").loc[PRIMARY]
        rule1, rule2 = p["psr"] >= 0.95, p["dsr"] >= 0.95
        rule3 = p["diff_ci_low"] > 0 or p["diff_ci_high"] < 0
        better = p["diff_vs_ou"] > 0
        if rule1 and rule3 and better:
            verdict = "beats OU after costs" + (" and survives multiple testing" if rule2 else "")
        elif rule1:
            verdict = "positive after costs, but not distinguishable from OU"
        elif rule3 and better:
            verdict = "better than OU, but no reliable positive edge"
        else:
            verdict = "no reliable edge after costs"
        lines = [
            f"# Phase 3 verdict ({period}, {years[0]}-{years[-1]})", "",
            f"Primary candidate: Transformer, seeds combined, net of {cfg['cost_bps']:g} bps.", "",
            f"- Net Sharpe {p['net_sharpe']:+.2f} (95% CI {p['ci_low']:+.2f} to {p['ci_high']:+.2f})",
            f"- Rule 1, PSR = {p['psr']:.3f} (needs >= 0.95): {'pass' if rule1 else 'fail'}",
            f"- Rule 2, DSR = {p['dsr']:.3f} with N = {n_trials} (needs >= 0.95): {'pass' if rule2 else 'fail'}",
            f"- Rule 3, Sharpe vs OU {p['diff_vs_ou']:+.2f} (95% CI {p['diff_ci_low']:+.2f} to "
            f"{p['diff_ci_high']:+.2f}, must exclude 0): {'pass' if rule3 else 'fail'}", "",
            f"**Verdict: {verdict}.**",
        ]
        (out / f"{period}_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n" + "\n".join(lines))
    print(f"\nwrote reports/phase3/{period}_*")


if __name__ == "__main__":
    main()
