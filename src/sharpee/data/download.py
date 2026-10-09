"""Download daily prices once and keep them as an immutable raw snapshot."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


def _fetch(tickers, start, end):
    """One yfinance call -> (close, volume), each dates x tickers. Prices are split/dividend adjusted."""
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, actions=False,
                     group_by="column", threads=True, progress=False)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    return df["Close"], df["Volume"]


def download_prices(tickers, start, end, chunk: int = 100):
    """Adjusted close and volume for all tickers, plus the tickers Yahoo could not supply.

    Delisted symbols are often missing from Yahoo; they are retried once and then
    reported, never silently dropped.
    """
    # yfinance's end date is exclusive
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    closes, volumes = [], []

    def run(batch_list):
        missing = []
        for i in range(0, len(batch_list), chunk):
            batch = batch_list[i:i + chunk]
            close, volume = _fetch(batch, start, end_excl)
            close = close.dropna(axis=1, how="all")
            closes.append(close)
            volumes.append(volume[close.columns])
            missing += [t for t in batch if t not in close.columns]
            print(f"  {i + len(batch)}/{len(batch_list)} requested, {len(missing)} missing so far")
        return missing

    failed = run(list(tickers))
    if failed:
        print(f"retrying {len(failed)} tickers")
        failed = run(failed)

    close = pd.concat(closes, axis=1)
    volume = pd.concat(volumes, axis=1)
    close = close.loc[:, ~close.columns.duplicated()].sort_index(axis=1)
    volume = volume.loc[:, ~volume.columns.duplicated()].reindex(columns=close.columns)
    return close, volume, sorted(failed)


def save_snapshot(raw_dir: Path, close, volume, membership, meta: dict) -> Path:
    """Write a new timestamped folder; existing snapshots are never overwritten."""
    out = Path(raw_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    close.to_parquet(out / "close.parquet")
    volume.to_parquet(out / "volume.parquet")
    membership.to_csv(out / "membership.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(meta, indent=2, default=str))
    return out


def latest_snapshot(raw_dir: Path) -> Path:
    snaps = sorted(p for p in Path(raw_dir).iterdir() if p.is_dir())
    if not snaps:
        raise FileNotFoundError(f"no raw snapshot in {raw_dir}; run scripts/download_data.py")
    return snaps[-1]
