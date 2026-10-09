"""Fit a scoring model on one fold with the portfolio-level Sharpe loss."""

import copy

import numpy as np
import torch

from ..backtest.costs import portfolio_returns
from ..backtest.engine import run_backtest
from ..evaluation.metrics import sharpe
from ..portfolio.construction import build_weights, to_global
from .dataset import DayBlocks
from .loss import sharpe_loss


def portfolio_forward(model, data: DayBlocks, sl: slice, cap, cost: float):
    """Scores -> weights -> net returns for one block, with gradients."""
    X, phi, tradable, universe, r_next = data.batch(sl)
    t, n, length = X.shape
    scores = model(X.reshape(t * n, length, 1)).view(t, n)
    x = build_weights(scores, phi, tradable, cap)
    x_glob = to_global(x, universe, data.n_tickers)
    _, _, net = portfolio_returns(x_glob, r_next, cost)
    return net, x_glob


@torch.no_grad()
def predict(model, data: DayBlocks, chunk: int = 64) -> np.ndarray:
    """Scores (D, N) for every date in data."""
    model.eval()
    out = []
    for a in range(0, len(data), chunk):
        X = data.X[a:a + chunk]
        t, n, length = X.shape
        out.append(model(X.reshape(t * n, length, 1)).view(t, n).float().cpu().numpy())
    return np.concatenate(out)


def validate(model, data: DayBlocks, cap, cost_bps: float):
    """Net returns over the full period, through the same backtester as the final evaluation."""
    frame = run_backtest(data.panel, data.idx, predict(model, data), cost_bps=cost_bps, cap=cap)
    return sharpe(frame["net"]), frame["net"]


def fit(model, train: DayBlocks, val: DayBlocks, cfg: dict, cap, cost_bps: float,
        seed: int = 0, log=print) -> dict:
    """Train with early stopping on validation net Sharpe; restores the best epoch's weights."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    train_cost = cfg.get("train_cost_bps", cost_bps) * 1e-4

    best, best_state, best_epoch, bad, history = -np.inf, None, -1, 0, []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        losses = []
        for sl in train.slices(cfg["block_days"], rng):
            net, x_glob = portfolio_forward(model, train, sl, cap, train_cost)
            # day 0 of a block trades from an unknown position, so it is left out
            loss = sharpe_loss(net[1:], x_glob[1:].sum(-1), cfg.get("exposure_lambda", 0.0))
            if not torch.isfinite(loss):  # e.g. a block where nothing is tradable
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            losses.append(loss.item())

        val_sharpe, _ = validate(model, val, cap, cost_bps)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_sharpe": val_sharpe})
        log(f"    epoch {epoch:3d}  train loss {np.mean(losses):+.4f}  val net Sharpe {val_sharpe:+.2f}")
        if np.isfinite(val_sharpe) and val_sharpe > best + 1e-6:
            best, best_epoch, bad = val_sharpe, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break

    if best_state is None:
        raise RuntimeError("no epoch produced a finite validation Sharpe; training diverged")
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_sharpe": best, "history": history}
