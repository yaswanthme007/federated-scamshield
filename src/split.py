"""Deterministic hash-based bank assignment and time-based train/test split."""

import hashlib

import numpy as np
import pandas as pd

from . import config


def _bank_boundaries():
    """Return cumulative bucket boundaries derived from BANK_RATIOS.

    With ratios [0.4, 0.3, 0.2, 0.1] and 100 buckets this gives [40, 70, 90, 100],
    i.e. buckets 0-39 -> bank 0, 40-69 -> bank 1, 70-89 -> bank 2, 90-99 -> bank 3.
    """
    cumulative = np.cumsum(config.BANK_RATIOS) * config.HASH_BUCKETS
    return np.round(cumulative).astype(int)


_BOUNDARIES = _bank_boundaries()


def account_bucket(account_id):
    """Map an account ID to a bucket 0..99 via the first 8 hex chars of its MD5."""
    digest = hashlib.md5(account_id.encode("utf-8")).hexdigest()
    return int(digest[: config.HASH_HEX_CHARS], 16) % config.HASH_BUCKETS


def account_bank(account_id):
    """Return the bank (0..NUM_BANKS-1) that owns ``account_id``."""
    return int(np.searchsorted(_BOUNDARIES, account_bucket(account_id), side="right"))


def assign_banks(names):
    """Vectorised bank assignment for a Series/array of account IDs.

    Args:
        names: iterable of account ID strings (e.g. ``df["nameOrig"]``).

    Returns:
        np.ndarray[int8] of bank IDs, aligned with ``names``.
    """
    buckets = np.fromiter(
        (account_bucket(n) for n in names), dtype=np.int16, count=len(names)
    )
    return np.searchsorted(_BOUNDARIES, buckets, side="right").astype(np.int8)


def add_bank_columns(df):
    """Add ``bank`` (sender's bank) and ``dest_bank`` columns in place and return df."""
    df["bank"] = assign_banks(df["nameOrig"].to_numpy())
    df["dest_bank"] = assign_banks(df["nameDest"].to_numpy())
    return df


def time_split_mask(df):
    """Boolean mask: True for training rows (step < TRAIN_TEST_SPLIT_STEP)."""
    return (df["step"] < config.TRAIN_TEST_SPLIT_STEP).to_numpy()


def split_by_bank(df):
    """Split a DataFrame into per-bank train/test partitions.

    Args:
        df: transactions with at least ``nameOrig`` and ``step`` columns. A
            ``bank`` column is added if missing.

    Returns:
        (banks, test_df) where ``banks`` is ``{bank_id: {"train": df, "test": df}}``
        and ``test_df`` is the combined (all-bank) test set used for global
        evaluation.
    """
    if "bank" not in df.columns:
        add_bank_columns(df)
    is_train = time_split_mask(df)
    banks = {}
    for bank_id in range(config.NUM_BANKS):
        in_bank = (df["bank"] == bank_id).to_numpy()
        banks[bank_id] = {
            "train": df[in_bank & is_train],
            "test": df[in_bank & ~is_train],
        }
    return banks, df[~is_train]


def describe_split(banks):
    """Print a per-bank summary of row and fraud counts."""
    print(f"  {'bank':<6}{'train rows':>12}{'train fraud':>13}{'test rows':>11}{'test fraud':>12}")
    for bank_id, parts in banks.items():
        tr, te = parts["train"], parts["test"]
        print(
            f"  {bank_id:<6}{len(tr):>12,}{int(tr[config.LABEL_COLUMN].sum()):>13,}"
            f"{len(te):>11,}{int(te[config.LABEL_COLUMN].sum()):>12,}"
        )


if __name__ == "__main__":
    sample = pd.DataFrame({
        "nameOrig": [f"C{i}" for i in range(10_000)],
        "nameDest": [f"C{i + 1}" for i in range(10_000)],
        "step": np.arange(10_000) % config.TOTAL_STEPS,
        config.LABEL_COLUMN: 0,
    })
    banks, test = split_by_bank(sample)
    shares = np.bincount(sample["bank"], minlength=config.NUM_BANKS) / len(sample)
    print("Bank shares on 10k synthetic IDs:", np.round(shares, 3))
    describe_split(banks)
