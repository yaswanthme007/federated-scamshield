"""Setup B: FedAvg across the 4 banks.

Each round the server broadcasts the global weights, every bank trains locally
on its own rows for LOCAL_EPOCHS, and the server replaces the global model with
the sample-weighted average  w_global = sum_k (n_k / n) * w_k.
Raw transactions never leave a bank; only weights do.
"""

import json
import time
from collections import OrderedDict

import torch

from . import config
from .evaluate import compute_metrics, format_metrics
from .model import (ScamMLP, get_model_weights, make_criterion, make_loader, make_optimizer,
                    predict_proba, set_model_weights, set_seed, train_one_epoch)


def fedavg(weight_list, sample_counts):
    """Sample-weighted average of client state_dicts.

    Floating-point tensors (weights, biases, BatchNorm running stats) are
    averaged with weights n_k / n. Integer buffers (BatchNorm's
    ``num_batches_tracked``) are taken from the largest client.
    """
    total = float(sum(sample_counts))
    largest = max(range(len(sample_counts)), key=lambda i: sample_counts[i])
    averaged = OrderedDict()
    for key in weight_list[0]:
        if weight_list[0][key].is_floating_point():
            averaged[key] = sum(w[key] * (n / total) for w, n in zip(weight_list, sample_counts))
        else:
            averaged[key] = weight_list[largest][key].clone()
    return averaged


def round_lr(round_idx):
    """Learning rate for a round, with optional linear warmup."""
    warmup = config.FED_LR_WARMUP_ROUNDS
    if warmup > 0 and round_idx < warmup:
        return config.LR * (round_idx + 1) / warmup
    return config.LR


def local_update(model, X, y, epochs, lr, seed, global_params=None):
    """Plain local training used by FedAvg/FedProx. Returns mean loss of last epoch."""
    optimizer = make_optimizer(model, lr=lr)
    criterion = make_criterion(y)
    loss = 0.0
    for epoch in range(epochs):
        loader = make_loader(X, y, shuffle=True, seed=seed + epoch)
        loss = train_one_epoch(model, loader, optimizer, criterion,
                               global_params=global_params, mu=config.FEDPROX_MU)
    return loss


def federated_train(data, rounds=config.FED_ROUNDS, local_epochs=config.LOCAL_EPOCHS,
                    local_update_fn=local_update, norm="batch", label="FedAvg"):
    """Generic FedAvg loop shared by Setup B and Setup E (DP).

    Args:
        data: PreparedData.
        rounds: number of communication rounds.
        local_epochs: epochs each bank trains per round.
        local_update_fn: ``fn(model, X, y, epochs, lr, seed, global_params) -> loss``.
        norm: ScamMLP normalisation ("batch" or "layer").
        label: name used in log lines.

    Returns:
        (global_model, per_round list of metric dicts, final test probabilities).
    """
    set_seed()
    global_model = ScamMLP(data.n_features, norm=norm)
    bank_data = [data.bank_train_data(b) for b in range(config.NUM_BANKS)]
    counts = [len(y) for _, y in bank_data]
    for b, (_, y) in enumerate(bank_data):
        print(f"  Bank {b}: {counts[b]:,} train rows ({counts[b] / sum(counts):.1%}), "
              f"{int(y.sum()):,} fraud")

    per_round, probs = [], None
    for r in range(rounds):
        t0 = time.time()
        lr = round_lr(r)
        global_weights = get_model_weights(global_model)
        client_weights, losses = [], []
        for b, (X, y) in enumerate(bank_data):
            client = ScamMLP(data.n_features, norm=norm)
            set_model_weights(client, global_weights)
            global_params = [p.detach().clone() for p in client.parameters()]
            loss = local_update_fn(client, X, y, local_epochs, lr,
                                   seed=config.RANDOM_SEED + 1000 * (r + 1) + 10 * b,
                                   global_params=global_params)
            client_weights.append(get_model_weights(client))
            losses.append(float(loss))
        set_model_weights(global_model, fedavg(client_weights, counts))

        probs = predict_proba(global_model, data.X_test)
        m = compute_metrics(data.y_test, probs)
        entry = {
            "round": r + 1,
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "recall_at_1pct_fpr": m["recall_at_1pct_fpr"],
            "f1": m["f1"],
            "lr": lr,
            "bank_losses": losses,
            "seconds": round(time.time() - t0, 1),
        }
        per_round.append(entry)
        loss_str = " ".join(f"b{b}={l:.4f}" for b, l in enumerate(losses))
        print(f"  [{label}] round {r + 1}/{rounds}  {format_metrics(m)}  "
              f"local loss: {loss_str}  ({entry['seconds']}s)")
    return global_model, per_round, probs


def run(data=None):
    """Run Setup B, save the global model and per-round metrics.

    Returns:
        (global_model, metrics dict with ``per_round``, test probabilities).
    """
    if data is None:
        from .features import prepare_datasets
        data = prepare_datasets()
    if config.FEDPROX_MU > 0:
        print(f"  FedProx enabled (mu={config.FEDPROX_MU})")
    model, per_round, probs = federated_train(data)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_FED_PATH)
    with open(config.FED_ROUNDS_PATH, "w") as f:
        json.dump(per_round, f, indent=2)

    metrics = compute_metrics(data.y_test, probs)
    metrics["per_round"] = per_round
    print(f"  Setup B final: {format_metrics(metrics)}")
    return model, metrics, probs


if __name__ == "__main__":
    run()
