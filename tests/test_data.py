import numpy as np
import pandas as pd

from sharpee.data.preprocess import MarketData, clean, eligibility
from sharpee.data.universe import membership_mask, tickers_between, to_yahoo


def toy_market(n_days=60):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    close = pd.DataFrame({"A": 50.0, "B": 40.0, "C": 3.0, "D": 30.0}, index=dates)
    close = close * np.linspace(1, 1.1, n_days)[:, None]
    volume = pd.DataFrame({"A": 1e6, "B": 2e6, "C": 1e7, "D": 1e5}, index=dates)
    member = pd.DataFrame(True, index=dates, columns=close.columns)
    member.loc[:, "D"] = False
    return MarketData(close, volume, member, pd.Series(0.0, index=dates))


def test_simple_returns():
    md = toy_market()
    expected = md.close.iloc[1] / md.close.iloc[0] - 1
    assert np.allclose(md.returns.iloc[1], expected)
    assert md.returns.iloc[0].isna().all()


def test_eligibility_rules():
    md = toy_market()
    e = eligibility(md, universe_size=1, min_price=5.0, history_days=20,
                    dollar_volume_window=5, min_history_frac=1.0)
    last = e.iloc[-1]
    assert not last["C"]          # price below the floor
    assert not last["D"]          # not an index member
    # A: 50 x 1e6 = 5e7 dollar volume, B: 40 x 2e6 = 8e7 -> B wins the single slot
    assert last["B"] and not last["A"]
    # not enough history at the start
    assert not e.iloc[:20].any().any()


def test_eligibility_ignores_the_future():
    md = toy_market()
    e1 = eligibility(md, 2, 5.0, 20, 5, 1.0)
    later = md.close.copy()
    later.iloc[40:] *= 0.01  # crash every price after day 40
    e2 = eligibility(MarketData(later, md.volume, md.member, md.market), 2, 5.0, 20, 5, 1.0)
    pd.testing.assert_frame_equal(e1.iloc[:40], e2.iloc[:40])


def test_clean_aligns_and_logs():
    dates = pd.bdate_range("2021-01-01", periods=5)
    close = pd.DataFrame({"SPY": [1, 2, 3, 4, 5.0], "X": [10, -1, 12, 13, 14.0], "Y": np.nan},
                         index=dates)
    close = pd.concat([close, close.iloc[[2]]])  # duplicate date
    volume = close * 0 + 100
    membership = pd.DataFrame({"ticker": ["X"], "start_date": [dates[0]], "end_date": [pd.NaT]})
    md, log = clean(close, volume, membership, "SPY")
    assert log["duplicate_dates"] == [str(dates[2].date())]
    assert log["nonpositive_prices"] == 1 and np.isnan(md.close.loc[dates[1], "X"])
    assert log["tickers_without_data"] == ["Y"]
    assert list(md.close.columns) == ["X"] and md.member["X"].all()


def test_membership_intervals():
    m = pd.DataFrame({"ticker": ["AAA", "AAA", "BRK.B"],
                      "start_date": pd.to_datetime(["2020-01-01", "2020-03-01", "2020-01-01"]),
                      "end_date": pd.to_datetime(["2020-01-31", pd.NaT, "2020-01-15"])})
    dates = pd.date_range("2020-01-01", "2020-04-01")
    mask = membership_mask(m, dates, ["AAA", "BRK-B"])
    assert mask.loc["2020-01-10", "AAA"] and not mask.loc["2020-02-10", "AAA"]
    assert mask.loc["2020-03-10", "AAA"]
    assert mask.loc["2020-01-10", "BRK-B"] and not mask.loc["2020-01-20", "BRK-B"]
    assert to_yahoo("BF.B") == "BF-B"
    assert tickers_between(m, "2020-02-01", "2020-02-28") == []
