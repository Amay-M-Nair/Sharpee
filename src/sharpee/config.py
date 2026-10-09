"""YAML configs live in <repo>/configs; paths in them are relative to the repo root."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(*names: str) -> dict:
    """Merge configs/<name>.yaml files in order; later files win on key clashes."""
    cfg = {}
    for name in names:
        with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


def repo_path(p) -> Path:
    """Resolve a config path against the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
