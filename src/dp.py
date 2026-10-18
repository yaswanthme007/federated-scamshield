"""Setup E: FedAvg with differentially private local training (DP-SGD).

Inside every bank, each batch's gradient is computed per sample, every
per-sample gradient is clipped to L2 norm MAX_GRAD_NORM, the clipped
gradients are summed, Gaussian noise N(0, (NOISE_MULTIPLIER * MAX_GRAD_NORM)^2)
is added, and the result is divided by the batch size. Only this noised
update ever shapes the weights a bank sends to the server.

BatchNorm mixes samples within a batch, which makes per-sample gradients (and
the DP guarantee) ill-defined, so the DP model uses LayerNorm instead.
"""

import json
import math

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

from . import config
from .evaluate import compute_metrics, format_metrics
from .model import make_criterion, make_loader, make_optimizer
from .train_fed import federated_train


def _per_sample_grads(model, params, buffers, criterion, xb, yb):
    """Per-sample gradients of the loss w.r.t. every parameter, via vmap(grad)."""

    def sample_loss(p, b, x, t):
        """Loss for one sample (adds and removes the batch dim)."""
        logit = functional_call(model, (p, b), (x.unsqueeze(0),)).squeeze(1)
        return criterion(logit, t.unsqueeze(0))

    return vmap(grad(sample_loss), in_dims=(None, None, 0, 0), randomness="different")(
        params, buffers, xb, yb
    )


def dp_step(model, optimizer, criterion, xb, yb, generator,
            max_grad_norm=config.MAX_GRAD_NORM, noise_multiplier=config.NOISE_MULTIPLIER):
    """One DP-SGD step: per-sample clip -> sum -> add Gaussian noise -> average -> step.

    Returns:
        mean (unclipped) batch loss, for logging.
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    grads = _per_sample_grads(model, params, buffers, criterion, xb, yb)

    batch = xb.shape[0]
    sq_norms = sum(g.reshape(batch, -1).pow(2).sum(1) for g in grads.values())
    clip = (max_grad_norm / (sq_norms.sqrt() + 1e-6)).clamp(max=1.0)

    optimizer.zero_grad()
    for name, p in model.named_parameters():
        g = grads[name]
        summed = torch.einsum("b,b...->...", clip, g)
        noise = torch.normal(0.0, noise_multiplier * max_grad_norm, size=summed.shape,
                             generator=generator)
        p.grad = (summed + noise) / batch
    optimizer.step()

    with torch.no_grad():
        return criterion(model(xb).squeeze(1), yb).item()


def dp_local_update(model, X, y, epochs, lr, seed, global_params=None):
    """DP-SGD replacement for ``train_fed.local_update``. Returns mean loss of last epoch."""
    optimizer = make_optimizer(model, lr=lr)
    criterion = make_criterion(y)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    loss = 0.0
    for epoch in range(epochs):
        losses = [dp_step(model, optimizer, criterion, xb, yb, generator)
                  for xb, yb in make_loader(X, y, shuffle=True, seed=seed + epoch)]
        loss = float(np.mean(losses)) if losses else 0.0
    return loss


def _log_a_int(q, sigma, alpha):
    """log A_alpha for the sampled Gaussian mechanism at integer order alpha (Mironov et al.)."""
    terms = [
        math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        + k * math.log(q) + (alpha - k) * math.log(1 - q)
        + (k * k - k) / (2 * sigma ** 2)
        for k in range(alpha + 1)
    ]
    peak = max(terms)
    return peak + math.log(sum(math.exp(t - peak) for t in terms))


def estimate_epsilon(sample_rate, steps, noise_multiplier=config.NOISE_MULTIPLIER,
                     delta=config.DP_DELTA, orders=config.DP_RDP_ORDERS):
    """(epsilon, best order) via an RDP accountant for the subsampled Gaussian.

    Assumes Poisson subsampling at rate ``sample_rate``; we use shuffled fixed
    batches, so treat the value as an estimate rather than a formal guarantee.
    """
    best = (math.inf, None)
    for a in orders:
        rdp = steps * _log_a_int(sample_rate, noise_multiplier, a) / (a - 1)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        if eps < best[0]:
            best = (eps, a)
    return best


def privacy_report(data, rounds=config.FED_ROUNDS, local_epochs=config.LOCAL_EPOCHS):
    """Per-bank epsilon estimates. The reported figure is the worst (largest) bank."""
    report = []
    for b in range(config.NUM_BANKS):
        n = int((data.bank_train == b).sum())
        q = min(config.BATCH_SIZE / n, 1.0)
        steps = math.ceil(n / config.BATCH_SIZE) * local_epochs * rounds
        eps, order = estimate_epsilon(q, steps)
        report.append({"bank_id": b, "sample_rate": q, "steps": steps,
                       "epsilon": eps, "rdp_order": order})
    return report


def run(data=None):
    """Run Setup E (FedAvg + DP-SGD), save model and per-round metrics.

    Returns:
        (global_model, metrics dict, test probabilities).
    """
    if data is None:
        from .features import prepare_datasets
        data = prepare_datasets()
    print(f"  DP-SGD: max_grad_norm={config.MAX_GRAD_NORM}, "
          f"noise_multiplier={config.NOISE_MULTIPLIER}, delta={config.DP_DELTA}, norm={config.DP_NORM}")
    model, per_round, probs = federated_train(data, local_update_fn=dp_local_update,
                                              norm=config.DP_NORM, label="FedAvg+DP")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_FED_DP_PATH)
    with open(config.FED_DP_ROUNDS_PATH, "w") as f:
        json.dump(per_round, f, indent=2)

    report = privacy_report(data)
    for r in report:
        print(f"  Bank {r['bank_id']}: q={r['sample_rate']:.5f}, steps={r['steps']}, "
              f"epsilon={r['epsilon']:.2f} (delta={config.DP_DELTA})")
    metrics = compute_metrics(data.y_test, probs)
    metrics.update(per_round=per_round,
                   dp_epsilon_estimate=max(r["epsilon"] for r in report),
                   dp_delta=config.DP_DELTA,
                   dp_per_bank=report,
                   noise_multiplier=config.NOISE_MULTIPLIER,
                   max_grad_norm=config.MAX_GRAD_NORM)
    print(f"  Setup E final: {format_metrics(metrics)}  epsilon~{metrics['dp_epsilon_estimate']:.2f}")
    return model, metrics, probs


if __name__ == "__main__":
    run()
