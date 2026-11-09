"""Time splits on unique days. Rows are never shuffled."""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def day_tss_splits(day: np.ndarray, n_splits: int = 5):
    """sklearn TimeSeriesSplit applied to the sorted unique days.

    Yields (fold, train_mask, val_mask, train_days, val_days).
    """
    udays = np.sort(np.unique(day))
    for k, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(udays), 1):
        tr_d, va_d = udays[tr], udays[va]
        yield k, np.isin(day, tr_d), np.isin(day, va_d), tr_d, va_d


def last_block_split(day: np.ndarray, horizon: int):
    """One split that looks like the real test: train on the past,
    validate on the last `horizon` days in one block."""
    udays = np.sort(np.unique(day))
    va_d = udays[-horizon:]
    tr_d = udays[:-horizon]
    return np.isin(day, tr_d), np.isin(day, va_d), tr_d, va_d


def origin_split(day: np.ndarray, train_end: int):
    """Train on days <= train_end, validate on every later day."""
    udays = np.sort(np.unique(day))
    tr_d, va_d = udays[udays <= train_end], udays[udays > train_end]
    return day <= train_end, day > train_end, tr_d, va_d
