"""Power analysis: what Sharpe ratios and differences could this design detect at all?

    python scripts/power_analysis.py                     # 150-stock results in reports/phase3
    SHARPEE_VARIANT=n400 python scripts/power_analysis.py   # 400-stock results in reports/n400/phase3

Reads the daily net returns saved by run_backtest.py (5 bps, seeds combined) and,
for each period, reports the minimum detectable effect (MDE) at 80% power:

- each strategy's Sharpe against zero, one-sided at 5% (the PSR >= 0.95 rule);
- the Transformer's Sharpe difference against every other strategy, two-sided
  at 5% (the paired-interval rules 3 and 4).

Standard errors come from the same stationary bootstrap as the confidence
intervals, so they keep fat tails, serial correlation and the correlation
between strategies. "Years needed" is how long the sample would have to be for
the observed effect to be detected with 80% power, with SE shrinking as
1/sqrt(length). Writes reports/.../phase3/power.csv and power.md.
"""

import numpy as np
import pandas as pd

from sharpee.config import load_config, repo_path
from sharpee.evaluation.significance import (bootstrap_sharpe_se, daily_sharpe, minimum_detectable,
                                             years_needed)

PRIMARY = "transformer"
NAMES = {"ou": "OU", "ridge": "Ridge", "gbm": "Gradient boosting", "mlp": "MLP",
         "temporal_cnn": "Temporal CNN", "transformer": "Transformer"}


def analyse(daily: pd.DataFrame, period: str) -> list[dict]:
    years = len(daily) / 252
    net = {s: daily[(s, "net")].to_numpy() for s in NAMES if (s, "net") in daily}
    rows = []
    for s, r in net.items():
        se = bootstrap_sharpe_se(r)
        sr = daily_sharpe(r) * np.sqrt(252)
        rows.append({"period": period, "test": "Sharpe vs 0", "strategy": NAMES[s], "observed": sr, "se": se,
                     "mde": minimum_detectable(se, two_sided=False),
                     "years_needed": years_needed(sr, se, years, two_sided=False) if sr > 0 else np.inf,
                     "corr": np.nan, "years": years})
    for s, r in net.items():
        if s == PRIMARY:
            continue
        a = net[PRIMARY]
        se = bootstrap_sharpe_se(a, r)
        diff = (daily_sharpe(a) - daily_sharpe(r)) * np.sqrt(252)
        rows.append({"period": period, "test": "Transformer minus", "strategy": NAMES[s], "observed": diff,
                     "se": se, "mde": minimum_detectable(se), "years_needed": years_needed(diff, se, years),
                     "corr": float(np.corrcoef(a, r)[0, 1]), "years": years})
    return rows


def fmt_years(y: float) -> str:
    return "never (observed ≤ 0)" if not np.isfinite(y) else f"{y:,.0f}"


def main():
    cfg = load_config("data", "experiment")
    out = repo_path(cfg["reports_dir"]) / "phase3"
    rows = []
    for period in ("test", "holdout"):
        path = out / f"{period}_daily.parquet"
        if path.exists():
            rows += analyse(pd.read_parquet(path), period)
    table = pd.DataFrame(rows)
    table.to_csv(out / "power.csv", index=False)

    lines = [f"# Power analysis ({cfg['universe_size']} stocks, net of {cfg['cost_bps']:g} bps)", "",
             "Minimum detectable effect (MDE) at 80% power, from stationary-bootstrap standard errors. "
             "Sharpe vs 0 is one-sided at 5%; differences are two-sided at 5%.", ""]
    for period, part in table.groupby("period", sort=False):
        lines += [f"## {'2020-2023' if period == 'test' else '2024-2025'} ({part['years'].iloc[0]:.1f} years)", "",
                  "| Test | Strategy | Observed | SE | MDE | Years needed | Daily corr. |", "|---|---|---|---|---|---|---|"]
        for r in part.itertuples():
            corr = "" if np.isnan(r.corr) else f"{r.corr:.2f}"
            lines.append(f"| {r.test} | {r.strategy} | {r.observed:+.2f} | {r.se:.2f} | {r.mde:.2f} | "
                         f"{fmt_years(r.years_needed)} | {corr} |")
        lines.append("")
    (out / "power.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
