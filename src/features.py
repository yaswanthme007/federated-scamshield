"""Feature engineering pipeline for PaySim.

Public entry points:
    load_raw()          -> raw TRANSFER/CASH_OUT DataFrame
    engineer_features() -> DataFrame with every engineered column added
    build_features()    -> (X_train, X_test, y_train, y_test), scaler saved
    prepare_datasets()  -> full pipeline (banks + rings + features + scaling)
"""

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config
from .split import add_bank_columns, time_split_mask

_RAW_DTYPES = {
    "step": np.int32,
    "type": "category",
    "amount": np.float64,
    "nameOrig": str,
    "oldbalanceOrg": np.float64,
    "newbalanceOrig": np.float64,
    "nameDest": str,
    "oldbalanceDest": np.float64,
    "newbalanceDest": np.float64,
    "isFraud": np.int8,
}


def load_raw(path=config.DATA_FILE):
    """Load PaySim, keep only TRANSFER/CASH_OUT rows and drop isFlaggedFraud.

    Args:
        path: CSV path (default data/PaySim.csv).

    Returns:
        DataFrame with the columns in ``config.RAW_COLUMNS``.
    """
    df = pd.read_csv(path, usecols=config.RAW_COLUMNS, dtype=_RAW_DTYPES)
    df = df[df["type"].isin(config.FRAUD_TYPES)].reset_index(drop=True)
    df["type"] = pd.Categorical(df["type"].astype(str), categories=config.FRAUD_TYPES)
    return df


def _transfer_to_cashout_flag(df):
    """1 for a TRANSFER whose recipient performs a CASH_OUT within the window.

    Classic mule pattern: money lands in an account and is cashed out shortly
    after. Pre-computed as a join between TRANSFER recipients and CASH_OUT
    senders, filtered to ``0 <= cashout_step - transfer_step <= window``.
    """
    flag = np.zeros(len(df), dtype=np.float32)
    is_tr = (df["type"] == "TRANSFER").to_numpy()
    is_co = ~is_tr
    transfers = pd.DataFrame({
        "idx": np.flatnonzero(is_tr),
        "acct": df["nameDest"].to_numpy()[is_tr],
        "step": df["step"].to_numpy()[is_tr],
    })
    cashouts = pd.DataFrame({
        "acct": df["nameOrig"].to_numpy()[is_co],
        "co_step": df["step"].to_numpy()[is_co],
    })
    joined = transfers.merge(cashouts, on="acct", how="inner")
    delay = joined["co_step"] - joined["step"]
    hits = joined.loc[(delay >= 0) & (delay <= config.CASHOUT_WINDOW_STEPS), "idx"]
    flag[hits.to_numpy()] = 1.0
    return flag


def _amount_matched_pair(df):
    """1 when a TRANSFER and a CASH_OUT of the identical amount occur close in time.

    Fraudsters move stolen funds and cash out the exact same sum within the
    hour. Both the TRANSFER and the matching CASH_OUT are flagged.
    """
    flag = np.zeros(len(df), dtype=np.float32)
    cents = np.round(df["amount"].to_numpy() * 100).astype(np.int64)
    step = df["step"].to_numpy().astype(np.int64)
    is_tr = (df["type"] == "TRANSFER").to_numpy()
    tr = pd.DataFrame({"tr_idx": np.flatnonzero(is_tr), "cents": cents[is_tr], "step": step[is_tr]})
    co = pd.DataFrame({"co_idx": np.flatnonzero(~is_tr), "cents": cents[~is_tr], "step": step[~is_tr]})
    for delay in range(config.AMOUNT_MATCH_WINDOW_STEPS + 1):
        shifted = tr.assign(step=tr["step"] + delay)
        joined = shifted.merge(co, on=["cents", "step"], how="inner")
        flag[joined["tr_idx"].to_numpy()] = 1.0
        flag[joined["co_idx"].to_numpy()] = 1.0
    return flag


def _dest_txn_count_window(df):
    """log1p(number of earlier transactions to the same nameDest in the trailing window).

    Causal (only earlier transactions count) and stationary: a lifetime count
    would grow with time and make every test row look unlike training rows.
    """
    dest_code = pd.factorize(df["nameDest"])[0].astype(np.int64)
    step = df["step"].to_numpy().astype(np.int64)
    span = config.TOTAL_STEPS + config.DEST_COUNT_WINDOW_STEPS + 1
    keys = dest_code * span + step
    order = np.argsort(keys, kind="stable")          # by (dest, step), ties in file order
    sorted_keys = keys[order]
    lower = np.searchsorted(sorted_keys, sorted_keys - config.DEST_COUNT_WINDOW_STEPS, side="left")
    counts = np.empty(len(df), dtype=np.float32)
    counts[order] = np.arange(len(df)) - lower
    return np.log1p(counts)


def engineer_features(df):
    """Add every engineered feature column to ``df`` (returns a new DataFrame).

    Args:
        df: raw TRANSFER/CASH_OUT transactions (may include injected rings).

    Returns:
        DataFrame with the original columns plus ``config.ALL_FEATURE_COLUMNS``.
    """
    out = df.copy()
    amount = out["amount"].to_numpy(dtype=np.float64)
    old_org = out["oldbalanceOrg"].to_numpy(dtype=np.float64)
    new_org = out["newbalanceOrig"].to_numpy(dtype=np.float64)
    old_dest = out["oldbalanceDest"].to_numpy(dtype=np.float64)
    new_dest = out["newbalanceDest"].to_numpy(dtype=np.float64)
    step = out["step"].to_numpy()

    out["amount_ratio"] = np.clip(amount / (old_org + 1), *config.AMOUNT_RATIO_CLIP)
    out["balance_delta_org"] = old_org - new_org
    out["balance_delta_dest"] = new_dest - old_dest
    out["orig_balance_zero"] = (old_org == 0).astype(np.float32)
    out["dest_balance_zero"] = (old_dest == 0).astype(np.float32)
    out["amount_equals_balance"] = (
        np.abs(amount - old_org) <= config.AMOUNT_MATCH_TOLERANCE
    ).astype(np.float32)
    out["is_whole_amount"] = (np.mod(amount, config.WHOLE_AMOUNT_MODULUS) == 0).astype(np.float32)
    hour = step % config.STEPS_PER_DAY
    out["hour_of_day"] = hour
    out["day_of_month"] = step // config.STEPS_PER_DAY
    out["hour_sin"] = np.sin(2 * np.pi * hour / config.STEPS_PER_DAY)
    out["hour_cos"] = np.cos(2 * np.pi * hour / config.STEPS_PER_DAY)
    out["amount_log"] = np.log1p(amount)
    out["balance_ratio_dest"] = np.clip(new_dest / (old_dest + 1), *config.BALANCE_RATIO_DEST_CLIP)
    out["transfer_to_cashout_flag"] = _transfer_to_cashout_flag(out)
    out["error_balance_org"] = new_org + amount - old_org
    out["error_balance_dest"] = old_dest + amount - new_dest
    out["type_encoded"] = out["type"].astype(str).map(config.TYPE_ENCODING).astype(np.float32)
    out["amount_matched_pair"] = _amount_matched_pair(out)
    out["dest_txn_count_24h"] = _dest_txn_count_window(out)
    return out


def to_model_matrix(df_feat, feature_columns=None):
    """Select model features and apply the signed-log transform to money columns.

    Returns:
        float32 array of shape (n_rows, n_features), not yet standardised.
    """
    feature_columns = feature_columns or config.FEATURE_COLUMNS
    X = df_feat[feature_columns].to_numpy(dtype=np.float64)
    if config.LOG_TRANSFORM_MONETARY:
        cols = [i for i, c in enumerate(feature_columns) if c in config.MONETARY_COLUMNS]
        X[:, cols] = np.sign(X[:, cols]) * np.log1p(np.abs(X[:, cols]))
    return X.astype(np.float32)


def fit_scaler(X_train, save=True):
    """Fit a StandardScaler on TRAINING data only and optionally persist it.

    Fitting on the union of all banks' training rows is equivalent to securely
    aggregating each bank's (count, sum, sum of squares) - no raw rows needed.
    """
    scaler = StandardScaler().fit(X_train)
    if save:
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, config.SCALER_PATH)
    return scaler


def apply_scaler(scaler, X):
    """Standardise ``X`` with a fitted scaler, clip outliers, return float32."""
    Z = scaler.transform(X)
    np.clip(Z, -config.SCALED_CLIP, config.SCALED_CLIP, out=Z)
    return Z.astype(np.float32)


def build_features(df):
    """Raw DataFrame -> (X_train, X_test, y_train, y_test); saves artifacts/scaler.pkl.

    Train/test is split by time (step < TRAIN_TEST_SPLIT_STEP is train). The
    scaler is fit on the training rows only.
    """
    feat = engineer_features(df)
    is_train = time_split_mask(feat)
    X = to_model_matrix(feat)
    y = feat[config.LABEL_COLUMN].to_numpy(dtype=np.float32)
    scaler = fit_scaler(X[is_train])
    return apply_scaler(scaler, X[is_train]), apply_scaler(scaler, X[~is_train]), y[is_train], y[~is_train]


@dataclass
class PreparedData:
    """Everything the training scripts need, produced by ``prepare_datasets``."""

    X_train: np.ndarray
    y_train: np.ndarray
    bank_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    bank_test: np.ndarray
    test_meta: pd.DataFrame
    feature_names: list
    ring_ground_truth: dict = field(default_factory=dict)

    @property
    def n_features(self):
        """Number of model input features."""
        return self.X_train.shape[1]

    def bank_train_data(self, bank_id):
        """Return (X, y) training arrays held by ``bank_id``."""
        mask = self.bank_train == bank_id
        return self.X_train[mask], self.y_train[mask]

    def bank_test_data(self, bank_id):
        """Return (X, y) test arrays held by ``bank_id``."""
        mask = self.bank_test == bank_id
        return self.X_test[mask], self.y_test[mask]


def prepare_datasets(path=config.DATA_FILE, inject=True):
    """Full data pipeline: load -> banks -> rings -> features -> split -> scale.

    Args:
        path: PaySim CSV path.
        inject: append synthetic mule rings to the test period.

    Returns:
        PreparedData.
    """
    from .inject_rings import inject_rings
    from .split import describe_split, split_by_bank

    t0 = time.time()
    print(f"  Loading {path} ...")
    df = load_raw(path)
    print(f"  {len(df):,} TRANSFER/CASH_OUT rows, {int(df[config.LABEL_COLUMN].sum()):,} fraud "
          f"({time.time() - t0:.1f}s)")

    add_bank_columns(df)
    ground_truth = {}
    if inject:
        df, ground_truth = inject_rings(df)
    else:
        df["ring_id"] = -1

    t1 = time.time()
    feat = engineer_features(df)
    del df
    print(f"  Engineered {len(config.FEATURE_COLUMNS)} model features ({time.time() - t1:.1f}s)")

    banks, _ = split_by_bank(feat)
    describe_split(banks)

    is_train = time_split_mask(feat)
    X = to_model_matrix(feat)
    y = feat[config.LABEL_COLUMN].to_numpy(dtype=np.float32)
    scaler = fit_scaler(X[is_train])
    meta_cols = ["step", "type", "amount", "nameOrig", "nameDest", "bank", "dest_bank",
                 "ring_id", config.LABEL_COLUMN]
    data = PreparedData(
        X_train=apply_scaler(scaler, X[is_train]),
        y_train=y[is_train],
        bank_train=feat["bank"].to_numpy()[is_train],
        X_test=apply_scaler(scaler, X[~is_train]),
        y_test=y[~is_train],
        bank_test=feat["bank"].to_numpy()[~is_train],
        test_meta=feat.loc[~is_train, meta_cols].reset_index(drop=True),
        feature_names=list(config.FEATURE_COLUMNS),
        ring_ground_truth=ground_truth,
    )
    pos_tr, pos_te = int(data.y_train.sum()), int(data.y_test.sum())
    print(f"  Train: {len(data.y_train):,} rows / {pos_tr:,} fraud ({pos_tr / len(data.y_train):.3%})")
    print(f"  Test : {len(data.y_test):,} rows / {pos_te:,} fraud ({pos_te / len(data.y_test):.3%})")
    print(f"  Data ready in {time.time() - t0:.1f}s")
    return data


if __name__ == "__main__":
    prepare_datasets()
