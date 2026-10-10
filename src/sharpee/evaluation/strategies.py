"""Phase 3 strategy books: OU positions and seed-combined walk-forward model scores."""

import json

import numpy as np
import pandas as pd

from ..strategies.ou_strategy import ou_positions

STRATEGIES = ["ou", "mlp", "temporal_cnn", "transformer", "transformer_nocost"]
COLUMNS = ["gross", "cost", "net", "turnover", "market", "net_exposure", "gross_exposure"]


def strategy_scores(panel, runs, period: str, idx: np.ndarray) -> dict:
    """{strategy: [(seed, scores aligned to idx)]}; models whose walk-forward runs are missing are skipped."""
    out = {}
    params = json.loads((runs / "best_ou.json").read_text())["params"]
    out["ou"] = [(0, ou_positions(panel, np.arange(len(panel.dates)), **params)[idx])]
    for name in STRATEGIES[1:]:
        seeds = []
        for seed in (0, 1, 2):
            f = runs / "walkforward" / period / f"{name}_seed{seed}.npz"
            if f.exists():
                z = np.load(f)
                if not np.array_equal(z["idx"], idx):
                    raise ValueError(f"{f.name} does not cover the {period} period exactly")
                seeds.append((seed, z["scores"]))
        if len(seeds) == 3:
            out[name] = seeds
        elif seeds:
            print(f"skipping {name}: {len(seeds)} of 3 seeds finished")
    return out


def combined(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Equal capital in each seed's book: the combined book's daily numbers are the seed averages."""
    return sum(f[COLUMNS] for f in frames) / len(frames)
