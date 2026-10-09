"""Small synthetic markets so the tests run in seconds without downloaded data."""

import pytest

from deepstat.data.synthetic import make_market
from deepstat.pipeline import panel_from_market

SMALL = {
    "universe_size": 30, "min_price": 5.0, "history_days": 140, "min_history_frac": 0.98,
    "dollar_volume_window": 20, "n_factors": 3, "pca_window": 100, "max_lookback": 40,
}


@pytest.fixture(scope="session")
def market():
    return make_market(n_stocks=40, n_days=600, n_factors=3, ar=0.9, seed=1)


@pytest.fixture(scope="session")
def panel(market):
    return panel_from_market(market, SMALL)
