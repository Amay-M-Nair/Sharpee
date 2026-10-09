"""Expanding walk-forward folds with embargoes.

Fold for test year Y: train [train_start, Y-val_years-1], validate the
val_years before Y, test Y. The last `embargo` days of the training and
validation periods are dropped, so no input window straddles two partitions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    name: str
    train: tuple[str, str]
    val: tuple[str, str]
    test: tuple[str, str]


def make_fold(test_start: int, test_end: int, train_start: int = 2010, val_years: int = 2) -> Fold:
    return Fold(
        name=f"test_{test_start}" if test_start == test_end else f"test_{test_start}_{test_end}",
        train=(f"{train_start}-01-01", f"{test_start - val_years - 1}-12-31"),
        val=(f"{test_start - val_years}-01-01", f"{test_start - 1}-12-31"),
        test=(f"{test_start}-01-01", f"{test_end}-12-31"),
    )


def make_folds(test_years, train_start: int = 2010, val_years: int = 2) -> list[Fold]:
    """One fold per test year; retrained yearly with an expanding training window."""
    return [make_fold(y, y, train_start, val_years) for y in test_years]


def split(panel, fold: Fold, embargo: int):
    """Date indices (train, val, test) for a fold, embargo applied."""
    train = panel.index_of(*fold.train)
    val = panel.index_of(*fold.val)
    test = panel.index_of(*fold.test)
    if embargo:
        train, val = train[:-embargo], val[:-embargo]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(f"{fold.name}: empty partition (train {len(train)}, val {len(val)}, test {len(test)})")
    assert train[-1] < val[0] and val[-1] < test[0]
    return np.asarray(train), np.asarray(val), np.asarray(test)
