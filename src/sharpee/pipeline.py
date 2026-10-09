"""Load processed market data and the residual panel, rebuilding the panel cache when stale."""

from pathlib import Path

from .config import repo_path
from .data.preprocess import MarketData, eligibility
from .features.pca_residuals import ResidualPanel, build_panel


def eligibility_from_config(md: MarketData, cfg: dict):
    return eligibility(md, cfg["universe_size"], cfg["min_price"], cfg["history_days"],
                       cfg["dollar_volume_window"], cfg["min_history_frac"])


def panel_from_market(md: MarketData, cfg: dict) -> ResidualPanel:
    return build_panel(md, eligibility_from_config(md, cfg), cfg["n_factors"], cfg["pca_window"],
                       cfg["max_lookback"], cfg["universe_size"])


def load_panel(cfg: dict, rebuild: bool = False) -> ResidualPanel:
    processed = repo_path(cfg["processed_dir"])
    if not (processed / "close.parquet").exists():
        raise FileNotFoundError(f"{processed} is empty; run scripts/download_data.py first")
    cache = processed / (f"panel_N{cfg['universe_size']}_K{cfg['n_factors']}"
                         f"_W{cfg['pca_window']}_L{cfg['max_lookback']}.npz")
    stale = not cache.exists() or cache.stat().st_mtime < (processed / "close.parquet").stat().st_mtime
    if rebuild or stale:
        panel = panel_from_market(MarketData.load(processed), cfg)
        panel.save(cache)
        return panel
    return ResidualPanel.load(cache)


def runs_dir(cfg: dict) -> Path:
    d = repo_path(cfg.get("runs_dir", "runs"))
    d.mkdir(parents=True, exist_ok=True)
    return d
