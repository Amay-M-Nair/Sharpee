"""Exposure diagnostics on stock weights in ticker space (T, G)."""

import numpy as np


def exposures(x_glob: np.ndarray) -> dict:
    """Net dollar, gross, long/short and largest single-name exposure per day."""
    return {
        "net_exposure": x_glob.sum(1),
        "gross_exposure": np.abs(x_glob).sum(1),
        "long_exposure": np.clip(x_glob, 0, None).sum(1),
        "short_exposure": np.clip(x_glob, None, 0).sum(1),
        "max_abs_weight": np.abs(x_glob).max(1),
        "n_positions": (np.abs(x_glob) > 1e-6).sum(1),
    }
