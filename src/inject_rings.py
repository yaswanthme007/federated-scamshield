"""Inject synthetic cross-bank mule rings into the test period.

Each ring is a chain of 3-6 accounts (A -> B -> C -> ...) moving 50K-500K via
TRANSFERs. Consecutive accounts are deliberately held at *different* banks, so
no single bank ever sees a whole ring - only the shared HMAC watchlist can
link them. Ring transactions are labelled fraud.
"""

import json

import numpy as np
import pandas as pd

from . import config
from .split import account_bank


def _new_account_id(rng, taken):
    """Draw a fresh customer ID that collides with no existing account."""
    while True:
        digits = rng.integers(0, 10 ** config.RING_ACCOUNT_DIGITS)
        name = f"{config.RING_ACCOUNT_PREFIX}{digits:0{config.RING_ACCOUNT_DIGITS}d}"
        if name not in taken:
            taken.add(name)
            return name


def _ring_accounts(rng, size, taken):
    """Create ``size`` accounts where each consecutive pair sits at different banks."""
    accounts = [_new_account_id(rng, taken)]
    while len(accounts) < size:
        candidate = _new_account_id(rng, taken)
        if account_bank(candidate) != account_bank(accounts[-1]):
            accounts.append(candidate)
        else:
            taken.discard(candidate)
    return accounts


def generate_rings(existing_accounts, seed=config.RANDOM_SEED):
    """Generate synthetic mule-ring transactions.

    Args:
        existing_accounts: set of account IDs already in the data (avoids collisions).
        seed: RNG seed.

    Returns:
        (ring_df, rings) where ``ring_df`` holds the raw-schema transactions and
        ``rings`` maps ring_id -> ordered list of member accounts.
    """
    rng = np.random.default_rng(seed)
    taken = set(existing_accounts)
    low, high = config.RING_AMOUNT_RANGE
    rows, rings = [], {}

    for ring_id in range(config.NUM_INJECTED_RINGS):
        size = int(rng.integers(config.RING_MIN_SIZE, config.RING_MAX_SIZE + 1))
        accounts = _ring_accounts(rng, size, taken)
        rings[ring_id] = accounts

        n_hops = size - 1
        latest_start = config.TOTAL_STEPS - 1 - n_hops * config.RING_MAX_HOP_DELAY_STEPS
        step = int(rng.integers(config.TRAIN_TEST_SPLIT_STEP, latest_start + 1))
        amount = float(rng.uniform(low, high))

        for hop in range(n_hops):
            sender, receiver = accounts[hop], accounts[hop + 1]
            if hop == 0:
                # Victim is drained in full by the scam payment.
                old_org = amount
            else:
                # Mule received the previous hop and forwards it minus a commission.
                old_org = amount
                amount = max(low, amount * (1 - rng.uniform(*config.RING_COMMISSION_RANGE)))
            amount = round(amount, 2)
            old_org = round(old_org, 2)
            unreported = rng.random() < config.RING_DEST_UNREPORTED_PROB
            rows.append({
                "step": step,
                "type": "TRANSFER",
                "amount": amount,
                "nameOrig": sender,
                "oldbalanceOrg": old_org,
                "newbalanceOrig": round(max(old_org - amount, 0.0), 2),
                "nameDest": receiver,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0 if unreported else amount,
                config.LABEL_COLUMN: 1,
                "ring_id": ring_id,
            })
            step += int(rng.integers(0, config.RING_MAX_HOP_DELAY_STEPS + 1))

    return pd.DataFrame(rows), rings


def inject_rings(df, seed=config.RANDOM_SEED, save=True):
    """Append synthetic mule rings to the test period of ``df``.

    Args:
        df: raw transactions (TRANSFER/CASH_OUT). If it has ``bank``/``dest_bank``
            columns, they are filled for the new rows too.
        seed: RNG seed.
        save: write the ground truth to results/ring_ground_truth.json.

    Returns:
        (df_with_rings, ring_ground_truth) where ``ring_ground_truth`` maps each
        ring account name to its ring ID. Original rows get ``ring_id = -1``.
    """
    existing = set(pd.unique(df["nameOrig"])) | set(pd.unique(df["nameDest"]))
    ring_df, rings = generate_rings(existing, seed=seed)
    if "bank" in df.columns:
        ring_df["bank"] = [account_bank(n) for n in ring_df["nameOrig"]]
        ring_df["dest_bank"] = [account_bank(n) for n in ring_df["nameDest"]]

    base = df.copy()
    base["ring_id"] = -1
    ring_df = ring_df.astype({c: base[c].dtype for c in ring_df.columns if c in base.columns})
    combined = pd.concat([base, ring_df], ignore_index=True)

    ground_truth = {acct: rid for rid, accts in rings.items() for acct in accts}
    n_cross = sum(len({account_bank(a) for a in accts}) > 1 for accts in rings.values())
    print(
        f"  Injected {len(rings)} rings ({len(ring_df)} transactions, "
        f"{len(ground_truth)} accounts); {n_cross}/{len(rings)} span >1 bank"
    )

    if save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "account_to_ring": ground_truth,
            "rings": {str(k): v for k, v in rings.items()},
            "ring_banks": {str(k): [account_bank(a) for a in v] for k, v in rings.items()},
        }
        with open(config.RING_GROUND_TRUTH_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    return combined, ground_truth


def load_ground_truth():
    """Load ``account -> ring_id`` from results/ring_ground_truth.json."""
    with open(config.RING_GROUND_TRUTH_PATH) as f:
        return {k: int(v) for k, v in json.load(f)["account_to_ring"].items()}


if __name__ == "__main__":
    demo = pd.DataFrame(columns=config.RAW_COLUMNS)
    _, gt = inject_rings(demo, save=False)
    print(f"  Example accounts: {list(gt.items())[:4]}")
