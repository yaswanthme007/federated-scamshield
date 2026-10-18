"""Setup C: centralised upper bound - one model on all banks' pooled data."""

import torch

from . import config
from .evaluate import compute_metrics, format_metrics
from .model import predict_proba, set_seed
from .train_local import train_bank_model


def run(data=None):
    """Train on the pooled training set, evaluate on the global test set.

    Returns:
        (model, metrics dict).
    """
    if data is None:
        from .features import prepare_datasets
        data = prepare_datasets()
    set_seed()
    print(f"  Pooled training set: {len(data.y_train):,} rows, {int(data.y_train.sum()):,} fraud")
    model = train_bank_model(data.X_train, data.y_train, data.n_features,
                             epochs=config.CENTRAL_EPOCHS, log_prefix="    [central] ")
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_CENTRAL_PATH)
    metrics = compute_metrics(data.y_test, predict_proba(model, data.X_test))
    print(f"  Setup C: {format_metrics(metrics)}")
    return model, metrics


if __name__ == "__main__":
    run()
