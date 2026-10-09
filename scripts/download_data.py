"""Download membership + prices into a new raw snapshot, then build data/processed.

    python scripts/download_data.py                  # download, then preprocess
    python scripts/download_data.py --skip-download  # re-run preprocessing on the latest snapshot
"""

import argparse
import json

import pandas as pd

from deepstat.config import load_config, repo_path
from deepstat.data.download import download_prices, latest_snapshot, save_snapshot
from deepstat.data.preprocess import clean, write_log
from deepstat.data.universe import fetch_membership, tickers_between


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    cfg = load_config("data")
    raw_dir, out_dir = repo_path(cfg["raw_dir"]), repo_path(cfg["processed_dir"])

    if not args.skip_download:
        membership = fetch_membership(cfg["membership_url"])
        tickers = tickers_between(membership, cfg["start"], cfg["end"])
        print(f"{len(tickers)} tickers were S&P 500 members between {cfg['start']} and {cfg['end']}")
        close, volume, failed = download_prices(tickers + [cfg["market_ticker"]], cfg["start"], cfg["end"])
        snap = save_snapshot(raw_dir, close, volume, membership, {
            "source": "yfinance", "membership_url": cfg["membership_url"],
            "start": cfg["start"], "end": cfg["end"],
            "requested": len(tickers) + 1, "received": close.shape[1], "failed": failed,
        })
        print(f"snapshot -> {snap}  ({close.shape[1]} received, {len(failed)} failed)")

    snap = latest_snapshot(raw_dir)
    close = pd.read_parquet(snap / "close.parquet")
    volume = pd.read_parquet(snap / "volume.parquet")
    membership = pd.read_csv(snap / "membership.csv", parse_dates=["start_date", "end_date"])
    md, log = clean(close, volume, membership, cfg["market_ticker"])
    log["snapshot"] = snap.name
    log["failed_downloads"] = json.loads((snap / "manifest.json").read_text())["failed"]
    md.save(out_dir)
    write_log(log, out_dir)
    print(f"processed -> {out_dir}: {md.close.shape[1]} tickers x {len(md.close)} days; "
          f"{len(log['failed_downloads'])} member tickers missing from Yahoo (see cleaning_log.json)")


if __name__ == "__main__":
    main()
