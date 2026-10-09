"""Validate and align raw prices, then build returns and the per-date eligibility mask."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import membership_mask


@dataclass
class MarketData:
    """Aligned daily panels on the market's trading calendar (dates x tickers)."""

    close: pd.DataFrame
    volume: pd.DataFrame
    member: pd.DataFrame   # bool, point-in-time index membership
    market: pd.Series      # benchmark simple returns (SPY)

    @property
    def returns(self) -> pd.DataFrame:
        """Simple returns R(t) = P(t) / P(t-1) - 1; NaN wherever either price is missing."""
        return self.close.pct_change(fill_method=None)

    def save(self, out: Path):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.close.to_parquet(out / "close.parquet")
        self.volume.to_parquet(out / "volume.parquet")
        self.member.to_parquet(out / "member.parquet")
        self.market.to_frame("market").to_parquet(out / "market.parquet")

    @classmethod
    def load(cls, src: Path) -> "MarketData":
        src = Path(src)
        return cls(
            close=pd.read_parquet(src / "close.parquet"),
            volume=pd.read_parquet(src / "volume.parquet"),
            member=pd.read_parquet(src / "member.parquet"),
            market=pd.read_parquet(src / "market.parquet")["market"],
        )


def clean(close: pd.DataFrame, volume: pd.DataFrame, membership: pd.DataFrame,
          market_ticker: str, jump: float = 0.5):
    """Raw snapshot -> (MarketData, log). Nothing is dropped without being recorded in the log."""
    log = {}
    close = close.sort_index()
    volume = volume.sort_index()

    dup = close.index.duplicated(keep="last")
    log["duplicate_dates"] = [str(d.date()) for d in close.index[dup]]
    close, volume = close[~dup], volume[~volume.index.duplicated(keep="last")]

    # the benchmark defines the trading calendar
    calendar = close[market_ticker].dropna().index
    market = close[market_ticker].reindex(calendar).pct_change(fill_method=None)
    stocks = [c for c in close.columns if c != market_ticker]
    close = close.reindex(index=calendar, columns=stocks)
    volume = volume.reindex(index=calendar, columns=stocks)
    log["dates_off_calendar_dropped"] = int(len(dup) - dup.sum() - len(calendar))

    bad = close <= 0
    log["nonpositive_prices"] = int(bad.to_numpy().sum())
    close = close.mask(bad)

    empty = close.columns[close.isna().all()]
    log["tickers_without_data"] = sorted(empty)
    close, volume = close.drop(columns=empty), volume.drop(columns=empty)

    r = close.pct_change(fill_method=None)
    jumps = (r.abs() > jump).sum()
    log["suspicious_jumps"] = {t: int(n) for t, n in jumps[jumps > 0].items()}

    member = membership_mask(membership, calendar, close.columns)
    log["n_tickers"] = int(close.shape[1])
    log["calendar"] = [str(calendar[0].date()), str(calendar[-1].date()), len(calendar)]
    return MarketData(close, volume.fillna(0.0), member, market), log


def eligibility(md: MarketData, universe_size: int, min_price: float, history_days: int,
                dollar_volume_window: int, min_history_frac: float = 0.98) -> pd.DataFrame:
    """(dates x tickers) bool: may this stock be in the universe on date t?

    Uses only data up to and including t: index member at t, enough trailing
    return history, price above the floor, and within the top `universe_size`
    by trailing median dollar volume. Stocks are never filtered on full-period
    history, which would quietly remove everything that was later delisted.
    """
    have = md.returns.notna().astype(np.float32)
    history = have.rolling(history_days, min_periods=1).sum() >= min_history_frac * history_days
    price_ok = md.close > min_price
    dollar_vol = (md.close * md.volume).rolling(dollar_volume_window,
                                                 min_periods=dollar_volume_window).median()
    base = md.member & history & price_ok & dollar_vol.notna()
    rank = dollar_vol.where(base).rank(axis=1, ascending=False, method="first")
    return base & (rank <= universe_size)


def write_log(log: dict, out: Path):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "cleaning_log.json").write_text(json.dumps(log, indent=2))
