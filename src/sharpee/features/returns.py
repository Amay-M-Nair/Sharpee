"""Return conventions. Simple returns everywhere; log returns only for diagnostics."""

import numpy as np
import pandas as pd


def simple_returns(close: pd.DataFrame) -> pd.DataFrame:
    """R(t) = P(t) / P(t-1) - 1. Portfolio returns are weighted sums of these."""
    return close.pct_change(fill_method=None)


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """For distribution plots only: log returns do not add across stocks."""
    return np.log(close / close.shift(1))
