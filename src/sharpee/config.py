"""YAML configs live in <repo>/configs; paths in them are relative to the repo root.

Setting the environment variable SHARPEE_VARIANT=<name> overlays
configs/variants/<name>.yaml on the data and experiment settings, so a
robustness run (e.g. a 400-stock universe) writes to its own runs and reports
folders and never touches the main results.
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(*names: str) -> dict:
    """Merge configs/<name>.yaml files in order; later files win on key clashes."""
    cfg = {}
    for name in names:
        cfg.update(_read(ROOT / "configs" / f"{name}.yaml"))
    variant = os.environ.get("SHARPEE_VARIANT")
    if variant and {"data", "experiment"} & set(names):
        cfg.update(_read(ROOT / "configs" / "variants" / f"{variant}.yaml"))
    return cfg


def repo_path(p) -> Path:
    """Resolve a config path against the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
