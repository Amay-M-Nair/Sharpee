"""Pipeline sanity check on synthetic markets with a known answer.

    planted mean reversion (AR(1) cumulative residuals) -> OU must earn a clearly positive Sharpe
    random-walk residuals                               -> Sharpe about 0 gross, negative net

If either fails, the bug is in the pipeline, not in the market.
"""

import numpy as np

from deepstat.backtest.engine import run_backtest
from deepstat.data.synthetic import make_market
from deepstat.evaluation.metrics import summarize
from deepstat.pipeline import panel_from_market
from deepstat.strategies.ou_strategy import ou_positions

CFG = {"universe_size": 100, "min_price": 5.0, "history_days": 312, "min_history_frac": 0.98,
       "dollar_volume_window": 20, "n_factors": 5, "pca_window": 252, "max_lookback": 60}


def run(ar, seed=0):
    md = make_market(n_stocks=100, n_days=1800, n_factors=5, ar=ar, seed=seed)
    panel = panel_from_market(md, CFG)
    idx = np.arange(len(panel.dates))
    pos = ou_positions(panel, idx)
    out = {}
    for cost in (0, 5):
        out[cost] = summarize(run_backtest(panel, idx, pos, cost_bps=cost))
    return out


def main():
    ok = True
    for name, ar, check in [
        ("mean-reverting residuals", 0.9, lambda s: s[0]["sharpe"] > 1.0),
        ("random-walk residuals", None, lambda s: abs(s[0]["sharpe"]) < 1.0 and s[5]["sharpe"] < s[0]["sharpe"]),
    ]:
        s = run(ar)
        passed = check(s)
        ok &= passed
        print(f"{name:26s} OU Sharpe gross {s[0]['sharpe']:+.2f}  net@5bps {s[5]['sharpe']:+.2f}  "
              f"beta {s[5]['beta']:+.2f}  -> {'PASS' if passed else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
