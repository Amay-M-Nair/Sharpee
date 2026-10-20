"""Load the exported result files and compute what the dashboard shows.

No Streamlit and no `sharpee` import here: the hosted app installs only
pandas, numpy, plotly and streamlit, and this module stays unit-testable.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The dashboard shows the 400-stock run (docs/robustness_universe_400.md); the 150-stock results are in the README
DASH = ROOT / "reports" / "n400" / "dashboard"
PHASE3 = ROOT / "reports" / "n400" / "phase3"
FIGURES = ROOT / "reports" / "figures"
PERIODS_PER_YEAR = 252

NAMES = {"ou": "OU", "ridge": "Ridge", "gbm": "Gradient boosting",
         "mlp": "MLP", "temporal_cnn": "Temporal CNN", "transformer": "Transformer",
         "transformer_nocost": "Transformer, no costs in loss"}
# High contrast on the dark theme; blue and orange stay distinct for colour-blind viewers
COLORS = {"ou": "#C9D1D9", "ridge": "#BC8CFF", "gbm": "#FF7B72", "mlp": "#58A6FF",
          "temporal_cnn": "#F0883E", "transformer": "#3DDC97", "transformer_nocost": "#F2CC60"}
BASELINES = {"ridge", "gbm"}   # non-neural baselines, drawn dotted
REFERENCE = "#6E7681"  # zero lines, thresholds
MAIN = ["ou", "ridge", "gbm", "mlp", "temporal_cnn", "transformer"]
DEFAULT = ["ou", "gbm", "mlp", "transformer"]   # what busy charts show unless the viewer picks more
PERIODS = {"test": "Test years 2020-2023", "holdout": "Holdout 2024-2025"}


@dataclass(frozen=True)
class Costs:
    """A cost model. Flat: bps per unit traded on every stock. Realistic (docs/realistic_costs.md): per-stock
    spread tiers times spread_mult, plus an annual borrow fee on short notional. Both are linear, so every
    number is exact for any setting."""
    bps: float = 5.0
    realistic: bool = False
    spread_mult: float = 1.0
    borrow: float = 0.005

    @property
    def label(self) -> str:
        if not self.realistic:
            return f"{self.bps:g} bps"
        return f"spread tiers ×{self.spread_mult:g} + {self.borrow:.1%} borrow"


def load_returns(period: str) -> pd.DataFrame:
    """Daily columns per strategy and delay d: '<s>.gross_d', '.turnover_d', '.spread_d' (tiered spread cost
    at x1), '.short_d' (short notional); plus '<s>.net_exposure' and 'market.return'."""
    return pd.read_parquet(DASH / f"returns_{period}.parquet").astype("float64")


def strategies_in(returns: pd.DataFrame) -> list[str]:
    present = {c.split(".")[0] for c in returns.columns}
    return [s for s in NAMES if s in present]


def net_returns(returns: pd.DataFrame, strategy: str, cost, delay: int = 0) -> pd.Series:
    """Net daily returns under a Costs model (or a flat cost in bps). Every cost is linear, so it is exact:
    flat: net = gross - bps * turnover; realistic: net = gross - mult * tiered spread - borrow / 252 * shorts."""
    if not isinstance(cost, Costs):
        cost = Costs(bps=float(cost))
    gross = returns[f"{strategy}.gross_d{delay}"]
    if not cost.realistic:
        return gross - cost.bps * 1e-4 * returns[f"{strategy}.turnover_d{delay}"]
    return (gross - cost.spread_mult * returns[f"{strategy}.spread_d{delay}"]
            - cost.borrow / PERIODS_PER_YEAR * returns[f"{strategy}.short_d{delay}"])


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


def summary(returns: pd.DataFrame, strategies: list[str], cost, delay: int = 0) -> pd.DataFrame:
    """One row per strategy at the given cost and execution delay."""
    market = returns["market.return"]
    rows = {}
    for s in strategies:
        r = net_returns(returns, s, cost, delay)
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


def sharpe_by_year(returns: pd.DataFrame, strategies: list[str], cost) -> pd.DataFrame:
    out = {}
    for s in strategies:
        r = net_returns(returns, s, cost)
        out[NAMES[s]] = r.groupby(r.index.year).apply(sharpe)
    return pd.DataFrame(out)


def sharpe_vs_cost(returns: pd.DataFrame, strategies: list[str], costs, delay: int = 0, index=None) -> pd.DataFrame:
    """Net Sharpe for each cost setting (Costs objects or flat bps), indexed by `index` or the costs."""
    costs = list(costs)
    return pd.DataFrame({NAMES[s]: [sharpe(net_returns(returns, s, c, delay)) for c in costs] for s in strategies},
                        index=list(index) if index is not None else costs)


def _phase3(realistic: bool) -> Path:
    """Official results: flat 5 bps, or the realistic-costs rerun (docs/realistic_costs.md)."""
    return PHASE3 / "realistic" if realistic else PHASE3


def load_significance(period: str, realistic: bool = False) -> pd.DataFrame:
    return pd.read_csv(_phase3(realistic) / f"{period}_significance.csv").set_index("strategy")


def load_neural_vs_baselines(period: str, realistic: bool = False) -> pd.DataFrame | None:
    """Paired Sharpe difference, Transformer minus each baseline (the amendment's rules 4 and 5)."""
    f = _phase3(realistic) / f"{period}_neural_vs_baselines.csv"
    return pd.read_csv(f).set_index("baseline") if f.exists() else None


def load_power(period: str) -> pd.DataFrame:
    """Minimum detectable effects at 80% power (scripts/power_analysis.py), net of 5 bps."""
    power = pd.read_csv(PHASE3 / "power.csv")
    return power[power["period"] == period].reset_index(drop=True)


def load_verdict(period: str, realistic: bool = False) -> str:
    return (_phase3(realistic) / f"{period}_verdict.md").read_text(encoding="utf-8")


def verdict_line(period: str, realistic: bool = False) -> str:
    """The bolded verdict sentence from the verdict file."""
    for line in load_verdict(period, realistic).splitlines():
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
    models = [m for m in ("ridge", "gbm", "mlp", "temporal_cnn", "transformer") if m in signals]
    out = signals.groupby(buckets, observed=True)[models].mean()
    out.index = out.index.astype(float)
    return out


def load_attention() -> pd.DataFrame:
    """Transformer attention received by each day of the 30-day window (test years, seed 0)."""
    return pd.read_csv(DASH / "attention_test.csv", index_col="day")
