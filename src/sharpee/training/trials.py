"""Trial log: every configuration evaluated on validation data, kept for the Deflated Sharpe Ratio.

Selecting the best of N configurations inflates its Sharpe; DSR corrects for
that using N and the spread of the trials' Sharpes. Both come from this log,
so nothing tried may be left out of it.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["trial_id", "logged_at", "model", "fold", "params", "val_start", "val_end",
           "n_days", "val_sharpe", "val_sr_daily"]


def log_trial(runs_dir: Path, model: str, fold: str, params: dict, val_net: pd.Series) -> str:
    runs_dir = Path(runs_dir)
    (runs_dir / "trials").mkdir(parents=True, exist_ok=True)
    trial_id = uuid.uuid4().hex[:10]
    r = val_net.to_numpy()
    sr = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0
    row = {
        "trial_id": trial_id, "logged_at": datetime.now().isoformat(timespec="seconds"),
        "model": model, "fold": fold, "params": json.dumps(params, sort_keys=True),
        "val_start": str(val_net.index[0].date()), "val_end": str(val_net.index[-1].date()),
        "n_days": len(r), "val_sharpe": sr * np.sqrt(252), "val_sr_daily": sr,
    }
    log = runs_dir / "trials.csv"
    pd.DataFrame([row], columns=COLUMNS).to_csv(log, mode="a", header=not log.exists(), index=False)
    val_net.rename("net").to_frame().to_parquet(runs_dir / "trials" / f"{trial_id}.parquet")
    return trial_id


def load_trials(runs_dir: Path) -> pd.DataFrame:
    log = Path(runs_dir) / "trials.csv"
    return pd.read_csv(log) if log.exists() else pd.DataFrame(columns=COLUMNS)
