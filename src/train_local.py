"""Setup A: each bank trains its own model on its own data only (local baseline)."""

import time

import numpy as np
import torch

from . import config
from .evaluate import compute_metrics, format_metrics
from .model import (ScamMLP, make_criterion, make_loader, make_optimizer, predict_proba,
                    set_seed, train_one_epoch)

AVERAGED_KEYS = ["pr_auc", "roc_auc", "recall_at_1pct_fpr", "f1"]


def train_bank_model(X, y, n_features, epochs=config.LOCAL_TRAIN_EPOCHS, seed=config.RANDOM_SEED,
                     log_prefix="  "):
    """Train a fresh ScamMLP on one bank's data and return it."""
    set_seed(seed)
    model = ScamMLP(n_features)
    optimizer = make_optimizer(model)
    criterion = make_criterion(y)
    for epoch in range(epochs):
        loader = make_loader(X, y, shuffle=True, seed=seed + epoch)
        loss = train_one_epoch(model, loader, optimizer, criterion)
        print(f"{log_prefix}epoch {epoch + 1:>2}/{epochs}  loss={loss:.4f}")
    return model


def run(data=None):
    """Train and evaluate one local model per bank on the GLOBAL test set.

    Returns:
        dict with averaged metrics plus a ``per_bank`` list.
    """
    if data is None:
        from .features import prepare_datasets
        data = prepare_datasets()
    set_seed()
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    per_bank = []
    for bank_id in range(config.NUM_BANKS):
        t0 = time.time()
        X, y = data.bank_train_data(bank_id)
        print(f"  Bank {bank_id}: {len(y):,} train rows, {int(y.sum()):,} fraud")
        model = train_bank_model(X, y, data.n_features, seed=config.RANDOM_SEED + bank_id,
                                 log_prefix=f"    [bank {bank_id}] ")
        torch.save(model.state_dict(), config.ARTIFACTS_DIR / config.MODEL_LOCAL_TEMPLATE.format(bank=bank_id))
        m = compute_metrics(data.y_test, predict_proba(model, data.X_test))
        m.update(bank_id=bank_id, train_rows=int(len(y)), train_fraud=int(y.sum()))
        per_bank.append(m)
        print(f"  Bank {bank_id} -> global test: {format_metrics(m)}  ({time.time() - t0:.1f}s)")

    result = {k: float(np.mean([m[k] for m in per_bank])) for k in AVERAGED_KEYS}
    result["pr_auc_min"] = float(min(m["pr_auc"] for m in per_bank))
    result["pr_auc_max"] = float(max(m["pr_auc"] for m in per_bank))
    result["per_bank"] = per_bank
    print(f"  Setup A average: {format_metrics(result)}")
    return result


if __name__ == "__main__":
    run()
