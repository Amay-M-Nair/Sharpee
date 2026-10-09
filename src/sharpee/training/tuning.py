"""Expand a model config's tuning grid and train one configuration."""

import copy
import itertools

import torch

from ..models import build_model
from .train import fit, validate


def expand(model_cfg: dict) -> list[dict]:
    """One full config per grid point. Grid keys are 'section.name', e.g. 'train.lr'.

    A config without a `tune` section expands to itself.
    """
    grid = model_cfg.get("tune") or {}
    keys = list(grid)
    base = {k: v for k, v in model_cfg.items() if k != "tune"}
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = copy.deepcopy(base)
        for key, value in zip(keys, values):
            section, name = key.split(".")
            cfg[section][name] = value
        out.append(cfg)
    return out


def train_config(cfg: dict, train, val, cap, cost_bps: float, seed: int, device, log=print):
    """Build, fit and validate one configuration. Returns (model, fit result, validation net returns)."""
    torch.manual_seed(seed)
    model = build_model(cfg["model"], cfg["lookback"], **cfg["params"]).to(device)
    result = fit(model, train, val, cfg["train"], cap, cost_bps, seed=seed, log=log)
    _, val_net = validate(model, val, cap, cost_bps)
    return model, result, val_net
