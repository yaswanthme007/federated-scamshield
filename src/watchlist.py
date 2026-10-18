"""Setup D: privacy-preserving cross-bank mule-ring detection.

1. Score every test transaction with the federated model.
2. Keep high-risk transactions (score > WATCHLIST_SCORE_THRESHOLD) as edges
   sender -> receiver of a directed money-flow graph. Their recipients are
   flagged, and money leaving a flagged account is traced onward (up to
   WATCHLIST_TRACE_HOPS hops) even if each forwarding payment looks benign on
   its own - that is how mules launder through clean-looking hops.
3. Connected components with >= RING_MIN_SIZE accounts that contain at least
   one pass-through account (receives AND forwards flagged money, i.e. a real
   A -> B -> C flow rather than a star of victims paying one account) are rings.
4. Ring members are HMAC-SHA256 hashed with the consortium key and written to
   SQLite, so banks can match accounts without exchanging raw IDs.
5. Ring recall is computed by hashing the injected ground truth with the same
   key and matching against what was stored.
"""

import hashlib
import hmac
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import config
from .split import assign_banks

SCHEMA = """
CREATE TABLE watchlist (
    account_hash TEXT,
    ring_id INT,
    bank_id INT,
    flag_date TEXT,
    score REAL,
    UNIQUE(account_hash, ring_id)
)
"""


def hmac_account(account_id, key=config.HMAC_KEY):
    """HMAC-SHA256 hex digest of an account ID under the shared consortium key."""
    return hmac.new(key.encode("utf-8"), account_id.encode("utf-8"), hashlib.sha256).hexdigest()


def step_to_date(step):
    """Map a simulation step (hours) to an ISO date string."""
    start = datetime.fromisoformat(config.SIM_START_DATE)
    return (start + timedelta(hours=int(step))).date().isoformat()


def trace_flagged_flows(scored, hops=config.WATCHLIST_TRACE_HOPS):
    """Select money-flow edges: high-risk transactions plus traced onward hops.

    Args:
        scored: transactions with ``nameOrig``, ``nameDest`` and ``score``.
        hops: how many times to follow money out of newly flagged accounts.

    Returns:
        Subset of ``scored`` forming the flagged-flow graph.
    """
    include = (scored["score"] > config.WATCHLIST_SCORE_THRESHOLD).to_numpy().copy()
    dest = scored["nameDest"].to_numpy()
    step = scored["step"].to_numpy()
    senders = scored["nameOrig"]

    def flag_times(mask):
        """Earliest step at which each recipient in ``mask`` received flagged money."""
        return pd.Series(step[mask]).groupby(dest[mask]).min()

    flagged_at = flag_times(include)
    for _ in range(hops):
        since = senders.map(flagged_at).to_numpy()          # NaN if sender not flagged
        new = ~include & (step >= since)                     # only money moved afterwards
        if not new.any():
            break
        include |= new
        flagged_at = flag_times(include)
    return scored[include]


def detect_rings(edges):
    """Find mule rings in a flagged-transaction graph.

    Args:
        edges: DataFrame with ``nameOrig``, ``nameDest``, ``step``, ``score``.

    Returns:
        DataFrame (one row per ring member) with columns
        ``account, ring_id, first_step, score``.
    """
    empty = pd.DataFrame(columns=["account", "ring_id", "first_step", "score"])
    if edges.empty:
        return empty
    codes, accounts = pd.factorize(pd.concat([edges["nameOrig"], edges["nameDest"]], ignore_index=True))
    src, dst = codes[: len(edges)], codes[len(edges):]
    n = len(accounts)
    graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    _, labels = connected_components(graph, directed=True, connection="weak")

    out_deg = np.bincount(src, minlength=n)
    in_deg = np.bincount(dst, minlength=n)
    pass_through = (out_deg > 0) & (in_deg > 0)
    sizes = np.bincount(labels)
    comps_with_flow = np.unique(labels[pass_through])
    keep = comps_with_flow[sizes[comps_with_flow] >= config.RING_MIN_SIZE]
    if len(keep) == 0:
        return empty

    both = np.concatenate([src, dst])
    node_score = pd.Series(np.tile(edges["score"].to_numpy(), 2)).groupby(both).max()
    node_step = pd.Series(np.tile(edges["step"].to_numpy(), 2)).groupby(both).min()
    members = np.flatnonzero(np.isin(labels, keep))
    ring_ids = pd.Series(labels[members]).rank(method="dense").astype(int).to_numpy() - 1
    return pd.DataFrame({
        "account": np.asarray(accounts)[members],
        "ring_id": ring_ids,
        "first_step": node_step.reindex(members).to_numpy(),
        "score": node_score.reindex(members).to_numpy(),
    })


def write_watchlist(rings, db_path):
    """(Re)create the SQLite watchlist and insert HMAC-hashed ring members."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS watchlist")
        conn.execute(SCHEMA)
        if not rings.empty:
            banks = assign_banks(rings["account"].to_numpy())
            rows = [
                (hmac_account(a), int(r), int(b), step_to_date(s), float(sc))
                for a, r, b, s, sc in zip(rings["account"], rings["ring_id"], banks,
                                          rings["first_step"], rings["score"])
            ]
            conn.executemany("INSERT OR IGNORE INTO watchlist VALUES (?, ?, ?, ?, ?)", rows)
        conn.commit()


def read_watchlist(db_path):
    """Load the watchlist table as a DataFrame."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM watchlist", conn)


def score_rings(stored, ground_truth, match_fraction=config.RING_MATCH_FRACTION):
    """Compare stored (hashed) rings with the injected ground truth.

    An injected ring counts as detected if one stored ring contains at least
    ``match_fraction`` of its members; it is *fully* reconstructed if one stored
    ring contains all of them.

    Args:
        stored: DataFrame with ``account_hash`` and ``ring_id``.
        ground_truth: dict raw account -> injected ring ID.

    Returns:
        dict of ring metrics.
    """
    truth = pd.DataFrame({"account_hash": [hmac_account(a) for a in ground_truth],
                          "true_ring": list(ground_truth.values())})
    true_sizes = truth.groupby("true_ring").size()
    hits = truth.merge(stored[["account_hash", "ring_id"]], on="account_hash", how="inner")
    overlap = hits.groupby(["true_ring", "ring_id"]).size().rename("overlap").reset_index()
    overlap["fraction"] = overlap["overlap"] / overlap["true_ring"].map(true_sizes)
    best = overlap.groupby("true_ring")["fraction"].max().reindex(true_sizes.index, fill_value=0.0)

    detected = int((best >= match_fraction).sum())
    full = int((best >= 1.0).sum())
    n_stored = int(stored["ring_id"].nunique()) if not stored.empty else 0
    matched_stored = overlap.loc[overlap["fraction"] >= match_fraction, "ring_id"].nunique()
    total = int(len(true_sizes))
    return {
        "ring_recall": detected / total if total else 0.0,
        "rings_detected": detected,
        "total_rings": total,
        "ring_full_recall": full / total if total else 0.0,
        "rings_fully_reconstructed": full,
        "detected_components": n_stored,
        "ring_precision": matched_stored / n_stored if n_stored else 0.0,
        "ring_accounts_flagged": int(hits["account_hash"].nunique()),
        "ring_accounts_total": int(len(truth)),
    }


def single_bank_ring_recall(scored, ground_truth):
    """Ring recall if each bank could only see transactions touching its own customers.

    Each bank flags and traces only within its own view, so it cannot follow
    money once it moves to another bank. Shows what the shared watchlist adds.
    """
    stored_all = []
    offset = 0
    for b in range(config.NUM_BANKS):
        visible = scored[(scored["bank"] == b) | (scored["dest_bank"] == b)]
        rings = detect_rings(trace_flagged_flows(visible))
        if rings.empty:
            continue
        rings = rings.assign(ring_id=rings["ring_id"] + offset)
        offset = int(rings["ring_id"].max()) + 1
        stored_all.append(pd.DataFrame({"account_hash": [hmac_account(a) for a in rings["account"]],
                                        "ring_id": rings["ring_id"]}))
    stored = pd.concat(stored_all, ignore_index=True) if stored_all else \
        pd.DataFrame(columns=["account_hash", "ring_id"])
    scores = score_rings(stored, ground_truth)
    return {"ring_recall": scores["ring_recall"], "ring_full_recall": scores["ring_full_recall"]}


def run(test_meta, probs, ground_truth, db_path=config.WATCHLIST_DB_PATH, label="D"):
    """Build the watchlist from model scores and evaluate ring detection.

    Args:
        test_meta: test-set DataFrame (nameOrig, nameDest, step, bank, dest_bank, ...).
        probs: fraud probabilities aligned with ``test_meta``.
        ground_truth: raw account -> injected ring ID.
        db_path: SQLite output path.
        label: setup name for logs.

    Returns:
        dict of ring-detection metrics.
    """
    scored = test_meta.assign(score=probs)
    n_high = int((scored["score"] > config.WATCHLIST_SCORE_THRESHOLD).sum())
    edges = trace_flagged_flows(scored)
    print(f"  [{label}] {n_high:,} / {len(scored):,} test transactions scored > "
          f"{config.WATCHLIST_SCORE_THRESHOLD}; {len(edges) - n_high:,} more traced from flagged accounts")
    rings = detect_rings(edges)
    write_watchlist(rings, db_path)
    stored = read_watchlist(db_path)
    metrics = score_rings(stored, ground_truth)
    metrics["high_risk_transactions"] = n_high
    metrics["flagged_transactions"] = int(len(edges))
    metrics["watchlist_rows"] = int(len(stored))
    metrics["cross_bank_components"] = int(
        stored.groupby("ring_id")["bank_id"].nunique().gt(1).sum()) if not stored.empty else 0
    metrics["single_bank_baseline"] = single_bank_ring_recall(scored, ground_truth)
    print(f"  [{label}] {metrics['detected_components']} candidate rings stored "
          f"({metrics['watchlist_rows']} hashed accounts, "
          f"{metrics['cross_bank_components']} span >1 bank) -> {db_path.name}")
    print(f"  [{label}] Ring recall {metrics['rings_detected']}/{metrics['total_rings']} "
          f"= {metrics['ring_recall']:.0%}  (fully reconstructed: {metrics['ring_full_recall']:.0%}, "
          f"precision: {metrics['ring_precision']:.0%})")
    sb = metrics["single_bank_baseline"]
    print(f"  [{label}] Siloed single-bank view: recall {sb['ring_recall']:.0%}, "
          f"fully reconstructed {sb['ring_full_recall']:.0%}")
    return metrics


if __name__ == "__main__":
    import torch

    from .features import prepare_datasets
    from .model import ScamMLP, predict_proba

    data = prepare_datasets()
    model = ScamMLP(data.n_features)
    model.load_state_dict(torch.load(config.MODEL_FED_PATH))
    run(data.test_meta, predict_proba(model, data.X_test), data.ring_ground_truth)
