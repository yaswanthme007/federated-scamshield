"""ScamMLP model definition plus shared training / evaluation helpers (CPU only)."""

import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from . import config


def set_seed(seed=config.RANDOM_SEED):
    """Seed python, numpy and torch, and pin torch to the configured CPU threads."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.TORCH_NUM_THREADS)


class FraudDataset(Dataset):
    """In-memory dataset over float32 numpy arrays.

    ``__getitem__`` accepts a single index or a list of indices; the latter lets
    ``make_loader`` fetch a whole batch with one fancy-index instead of 2048
    Python calls, which is ~50x faster on CPU.
    """

    def __init__(self, X, y):
        """Wrap ``X`` (n, d) and ``y`` (n,) as float32 tensors (zero-copy when possible)."""
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self):
        """Number of samples."""
        return len(self.y)

    def __getitem__(self, idx):
        """Return ``(X[idx], y[idx])`` for an int or a list of indices."""
        return self.X[idx], self.y[idx]


def make_loader(X, y, batch_size=config.BATCH_SIZE, shuffle=True, seed=config.RANDOM_SEED):
    """Build a batched DataLoader over numpy arrays.

    Args:
        X, y: numpy arrays.
        batch_size: rows per batch.
        shuffle: random order (training) vs sequential (evaluation).
        seed: seed for the shuffling generator.
    """
    dataset = FraudDataset(X, y)
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    batches = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    return DataLoader(dataset, sampler=batches, batch_size=None)


class ScamMLP(nn.Module):
    """Two-hidden-layer MLP emitting one fraud logit per transaction.

    Input -> Linear(H1) -> Norm -> ReLU -> Dropout
          -> Linear(H2) -> Norm -> ReLU -> Dropout -> Linear(1)

    ``norm="batch"`` (default) uses BatchNorm, which helps FedAvg converge.
    ``norm="layer"`` uses LayerNorm, required for per-sample DP gradients.
    """

    def __init__(self, n_features, hidden_1=config.HIDDEN_1, hidden_2=config.HIDDEN_2,
                 dropout=config.DROPOUT, norm="batch"):
        """Build the network for ``n_features`` inputs."""
        super().__init__()
        norm_layer = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm}[norm]
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_1),
            norm_layer(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            norm_layer(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, x):
        """Return raw logits of shape (batch, 1)."""
        return self.net(x)


def compute_pos_weight(y):
    """Class-imbalance weight for BCEWithLogitsLoss: neg/pos (or its sqrt)."""
    pos = float(np.sum(y))
    neg = float(len(y) - pos)
    ratio = neg / max(pos, 1.0)
    if config.POS_WEIGHT_MODE == "sqrt":
        ratio = ratio ** 0.5
    return ratio


def make_criterion(y):
    """BCEWithLogitsLoss with pos_weight derived from labels ``y``."""
    pos_weight = torch.tensor([compute_pos_weight(y)], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def make_optimizer(model, lr=config.LR):
    """Adam with the configured weight decay."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)


def train_one_epoch(model, dataloader, optimizer, criterion, global_params=None, mu=0.0):
    """Run one training epoch and return the mean batch loss.

    Args:
        model, dataloader, optimizer, criterion: the usual.
        global_params: list of tensors (the server model's parameters). When
            given with ``mu > 0`` the FedProx term ``mu/2 * ||w - w_global||^2``
            is added to the loss.
        mu: FedProx proximal coefficient.
    """
    model.train()
    total, n_batches = 0.0, 0
    for xb, yb in dataloader:
        if len(yb) < 2:          # BatchNorm cannot normalise a single sample
            continue
        optimizer.zero_grad()
        loss = criterion(model(xb).squeeze(1), yb)
        if global_params is not None and mu > 0:
            prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
            loss = loss + 0.5 * mu * prox
        loss.backward()
        optimizer.step()
        total += loss.item()
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader):
    """Score a dataloader. Returns (probabilities, labels) as numpy arrays.

    The sigmoid is taken in float64: in float32 it rounds every logit above
    ~17 to exactly 1.0, creating ties that silently deflate PR-AUC.
    """
    model.eval()
    probs, labels = [], []
    for xb, yb in dataloader:
        probs.append(torch.sigmoid(model(xb).squeeze(1).double()))
        labels.append(yb)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def predict_proba(model, X, batch_size=config.EVAL_BATCH_SIZE):
    """Convenience wrapper: fraud probabilities for a numpy feature matrix."""
    probs, _ = evaluate(model, make_loader(X, np.zeros(len(X), np.float32), batch_size, shuffle=False))
    return probs


def get_model_weights(model):
    """Return a detached copy of the model's state_dict as an OrderedDict."""
    return OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())


def set_model_weights(model, weights):
    """Load ``weights`` (an OrderedDict state_dict) into ``model``."""
    model.load_state_dict(weights)
