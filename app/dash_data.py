"""Load the exported result files and compute what the dashboard shows.

No Streamlit and no `sharpee` import here: the hosted app installs only
pandas, numpy, plotly and streamlit, and this module stays unit-testable.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "reports" / "dashboard"
PHASE3 = ROOT / "reports" / "phase3"
FIGURES = ROOT / "reports" / "figures"
PERIODS_PER_YEAR = 252

NAMES = {"ou": "OU", "mlp": "MLP", "temporal_cnn": "Temporal CNN", "transformer": "Transformer",
         "transformer_nocost": "Transformer, no costs in loss"}
# High contrast on the dark theme; blue and orange stay distinct for colour-blind viewers
COLORS = {"ou": "#C9D1D9", "mlp": "#58A6FF", "temporal_cnn": "#F0883E", "transformer": "#3DDC97",
          "transformer_nocost": "#F2CC60"}
REFERENCE = "#6E7681"  # zero lines, thresholds
MAIN = ["ou", "mlp", "temporal_cnn", "transformer"]
PERIODS = {"test": "Test years 2020-2023", "holdout": "Holdout 2024-2025"}


def load_returns(period: str) -> pd.DataFrame:
    """Daily columns '<strategy>.gross_d0', '.turnover_d0', '.gross_d1', '.turnover_d1', '.net_exposure', 'market.return'."""
    return pd.read_parquet(DASH / f"returns_{period}.parquet").astype("float64")


def strategies_in(returns: pd.DataFrame) -> list[str]:
    present = {c.split(".")[0] for c in returns.columns}
    return [s for s in NAMES if s in present]


def net_returns(returns: pd.DataFrame, strategy: str, cost_bps: float, delay: int = 0) -> pd.Series:
    """Costs are linear in traded notional, so any cost level is exact: net = gross - cost * turnover."""
    gross = returns[f"{strategy}.gross_d{delay}"]
    return gross - cost_bps * 1e-4 * returns[f"{strategy}.turnover_d{delay}"]


def sharpe(r: pd.Series) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else 0.0


def equity(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    eq = equity(r)
    return eq / eq.cummax() - 1


def rolling_beta(r: pd.Series, market: pd.Series, window: int = 63) -> pd.Series:
    return r.rolling(window).cov(market) / market.rolling(window).var()


def summary(returns: pd.DataFrame, strategies: list[str], cost_bps: float, delay: int = 0) -> pd.DataFrame:
    """One row per strategy at the given cost and execution delay."""
    market = returns["market.return"]
    rows = {}
    for s in strategies:
        r = net_returns(returns, s, cost_bps, delay)
        rows[NAMES[s]] = {
            "net Sharpe": sharpe(r),
            "gross Sharpe": sharpe(returns[f"{s}.gross_d{delay}"]),
            "ann. return": r.mean() * PERIODS_PER_YEAR,
            "ann. vol": r.std() * np.sqrt(PERIODS_PER_YEAR),
            "max drawdown": float(drawdown(r).min()),
            "turnover / day": returns[f"{s}.turnover_d{delay}"].mean(),
            "beta": float(r.cov(market) / market.var()),
        }
    return pd.DataFrame(rows).T


def sharpe_by_year(returns: pd.DataFrame, strategies: list[str], cost_bps: float) -> pd.DataFrame:
    out = {}
    for s in strategies:
        r = net_returns(returns, s, cost_bps)
        out[NAMES[s]] = r.groupby(r.index.year).apply(sharpe)
    return pd.DataFrame(out)


def sharpe_vs_cost(returns: pd.DataFrame, strategies: list[str], costs, delay: int = 0) -> pd.DataFrame:
    return pd.DataFrame({NAMES[s]: [sharpe(net_returns(returns, s, c, delay)) for c in costs] for s in strategies},
                        index=list(costs))


def load_significance(period: str) -> pd.DataFrame:
    return pd.read_csv(PHASE3 / f"{period}_significance.csv").set_index("strategy")


def load_verdict(period: str) -> str:
    return (PHASE3 / f"{period}_verdict.md").read_text(encoding="utf-8")


def verdict_line(period: str) -> str:
    """The bolded verdict sentence from the verdict file."""
    for line in load_verdict(period).splitlines():
        if line.startswith("**Verdict:"):
            return line.strip("*").replace("Verdict: ", "").rstrip(".")
    return ""


def load_signals(period: str) -> pd.DataFrame:
    return pd.read_parquet(DASH / f"signals_{period}.parquet")


def load_trials() -> pd.DataFrame:
    return pd.read_csv(DASH / "trials.csv")


def learned_rule(signals: pd.DataFrame, step: float = 0.5, limit: float = 3.0) -> pd.DataFrame:
    """Mean standardized model score per OU s-score bucket: what each network does at each stretch."""
    edges = np.arange(-limit, limit + step / 2, step)
    mids = (edges[:-1] + edges[1:]) / 2
    buckets = pd.cut(signals["s_score"], edges, labels=mids)
    models = [m for m in ("mlp", "temporal_cnn", "transformer") if m in signals]
    out = signals.groupby(buckets, observed=True)[models].mean()
    out.index = out.index.astype(float)
    return out


def load_attention() -> pd.DataFrame:
    """Transformer attention received by each day of the 30-day window (test years, seed 0)."""
    return pd.read_csv(DASH / "attention_test.csv", index_col="day")
