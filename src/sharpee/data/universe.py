"""Point-in-time S&P 500 membership from the public fja05680/sp500 dataset."""

import io

import pandas as pd
import requests


def fetch_membership(url: str) -> pd.DataFrame:
    """Membership intervals: one row per (ticker, start_date, end_date); NaT end = still a member."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), parse_dates=["start_date", "end_date"])


def to_yahoo(ticker: str) -> str:
    """BRK.B -> BRK-B: Yahoo uses '-' as the share-class separator."""
    return ticker.replace(".", "-")


def tickers_between(membership: pd.DataFrame, start, end) -> list[str]:
    """Every ticker that was a member at any point in [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    left = membership["end_date"].fillna(pd.Timestamp.max)
    keep = (membership["start_date"] <= end) & (left >= start)
    return sorted({to_yahoo(t) for t in membership.loc[keep, "ticker"]})


def membership_mask(membership: pd.DataFrame, dates: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    """(dates x tickers) bool: True where the ticker was an index member on that date.

    A ticker can appear in several intervals (it left and rejoined, or the symbol
    was reused); any covering interval counts.
    """
    mask = pd.DataFrame(False, index=dates, columns=list(tickers))
    for row in membership.itertuples(index=False):
        t = to_yahoo(row.ticker)
        if t not in mask.columns:
            continue
        end = row.end_date if pd.notna(row.end_date) else dates[-1]
        mask.loc[(dates >= row.start_date) & (dates <= end), t] = True
    return mask
