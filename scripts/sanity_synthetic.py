"""Pipeline sanity check on synthetic markets with a known answer.

    planted mean reversion (AR(1) cumulative residuals) -> every strategy must earn a clearly positive Sharpe
    random-walk residuals                               -> Sharpe about 0 gross, lower net

Neural models train on the first part of the sample, early-stop on the next
and are scored on the last part, so a lucky validation period cannot pass the
check. If a check fails, the bug is in the pipeline, not in the market.

    python scripts/sanity_synthetic.py            # OU + MLP + temporal CNN + Transformer
    python scripts/sanity_synthetic.py --ou-only  # seconds instead of minutes
"""

import argparse

import numpy as np
import torch

from sharpee.backtest.engine import run_backtest
from sharpee.config import load_config
from sharpee.data.synthetic import make_market
from sharpee.evaluation.metrics import sharpe
from sharpee.pipeline import panel_from_market
from sharpee.strategies.ou_strategy import ou_positions
from sharpee.training.dataset import DayBlocks
from sharpee.training.train import predict
from sharpee.training.tuning import train_config

CFG = {"universe_size": 100, "min_price": 5.0, "history_days": 312, "min_history_frac": 0.98,
       "dollar_volume_window": 20, "n_factors": 5, "pca_window": 252, "max_lookback": 60}
MODELS = ["mlp", "temporal_cnn", "transformer"]


def scores_for(name, panel, train, val, test, device):
    cfg = load_config(name)
    cfg.pop("tune", None)
    cfg["train"].update(max_epochs=15, patience=4)
    data = {k: DayBlocks(panel, idx, cfg["lookback"], device) for k, idx in
            {"train": train, "val": val, "test": test}.items()}
    model, _, _ = train_config(cfg, data["train"], data["val"], 0.02, 5.0, seed=0, device=device,
                               log=lambda _: None)
    return predict(model, data["test"])


def run(ar, with_models, device):
    panel = panel_from_market(make_market(n_stocks=100, n_days=2500, n_factors=5, ar=ar, seed=0), CFG)
    n = len(panel.dates)
    train, val, test = np.arange(0, 1100), np.arange(1130, 1530), np.arange(1560, n)
    ou = ou_positions(panel, np.arange(n))
    out = {"ou": ou[test]}
    if with_models:
        for name in MODELS:
            out[name] = scores_for(name, panel, train, val, test, device)
    # OU's +1/-1 positions are rule-based and not centered; model scores are
    return {k: tuple(sharpe(run_backtest(panel, test, s, cost_bps=c, center=k != "ou")["net"]) for c in (0, 5))
            for k, s in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ou-only", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ok = True
    for label, ar, check in [
        ("mean-reverting residuals", 0.9, lambda gross, net: net > 2.0),
        ("random-walk residuals", None, lambda gross, net: abs(gross) < 1.5 and net < gross),
    ]:
        print(label)
        for name, (gross, net) in run(ar, not args.ou_only, device).items():
            passed = check(gross, net)
            ok &= passed
            print(f"  {name:13s} Sharpe gross {gross:+6.2f}  net@5bps {net:+6.2f}  -> {'PASS' if passed else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
