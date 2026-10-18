"""Master orchestrator: data -> A -> C -> B -> D -> E -> metrics/plots -> ONNX.

Usage:  python -m src.run_all [--data path/to/PaySim.csv]
"""

import argparse
import time
from contextlib import contextmanager
from pathlib import Path

from . import config, dp, export_onnx, train_central, train_fed, train_local, watchlist
from .evaluate import make_all_plots, save_metrics
from .features import prepare_datasets
from .model import set_seed

TIMINGS = {}


@contextmanager
def stage(title, key):
    """Print a banner, time the block, and record the duration under ``key``."""
    print(f"\n=== {title} ===")
    t0 = time.time()
    yield
    TIMINGS[key] = round(time.time() - t0, 1)
    print(f"--- {title} done in {TIMINGS[key]:.1f}s ---")


def summary_table(metrics):
    """Render the final comparison table as a string."""
    rows = [
        ("A  Local (avg of 4 banks)", metrics["setup_A"]),
        ("B  Federated (FedAvg)", metrics["setup_B"]),
        ("C  Centralised (ceiling)", metrics["setup_C"]),
        ("E  Federated + DP", metrics["setup_E"]),
    ]
    header = f"{'Setup':<28}{'PR-AUC':>9}{'ROC-AUC':>9}{'R@1%FPR':>9}{'F1':>8}"
    lines = [header, "-" * len(header)]
    for name, m in rows:
        lines.append(f"{name:<28}{m['pr_auc']:>9.4f}{m['roc_auc']:>9.4f}"
                     f"{m['recall_at_1pct_fpr']:>9.4f}{m['f1']:>8.4f}")
    d, e = metrics["setup_D"], metrics["setup_E"]
    lines.append("-" * len(header))
    lines.append(f"D  Ring recall (B model): {d['rings_detected']}/{d['total_rings']} = {d['ring_recall']:.0%}"
                 f"  | fully reconstructed {d['ring_full_recall']:.0%}"
                 f"  | siloed banks {d['single_bank_baseline']['ring_recall']:.0%}")
    lines.append(f"E  Ring recall (DP model): {e['rings_detected']}/{e['total_rings']} = {e['ring_recall']:.0%}"
                 f"  | epsilon ~ {e['dp_epsilon_estimate']:.2f} (delta={e['dp_delta']})")
    c_gap = metrics["setup_C"]["pr_auc"] - metrics["setup_B"]["pr_auc"]
    lines.append(f"Federated gap to centralised ceiling: {c_gap:+.4f} PR-AUC")
    return "\n".join(lines)


def main(data_path=config.DATA_FILE):
    """Run every setup in order and save all results."""
    t_start = time.time()
    set_seed()
    for d in (config.RESULTS_DIR, config.ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    metrics = {}

    with stage("DATA: load, bank split, ring injection, features", "data"):
        data = prepare_datasets(data_path, inject=True)

    with stage("SETUP A: Local single-bank baselines", "setup_A"):
        metrics["setup_A"] = train_local.run(data)

    with stage("SETUP C: Centralised upper bound", "setup_C"):
        _, metrics["setup_C"] = train_central.run(data)

    with stage("SETUP B: Federated Training (FedAvg)", "setup_B"):
        fed_model, metrics["setup_B"], fed_probs = train_fed.run(data)

    with stage("SETUP D: Cross-bank mule-ring watchlist", "setup_D"):
        ring = watchlist.run(data.test_meta, fed_probs, data.ring_ground_truth,
                             db_path=config.WATCHLIST_DB_PATH, label="D")
        metrics["setup_D"] = {"pr_auc": metrics["setup_B"]["pr_auc"], **ring}

    with stage("SETUP E: Federated + Differential Privacy", "setup_E"):
        _, metrics["setup_E"], dp_probs = dp.run(data)
        ring_dp = watchlist.run(data.test_meta, dp_probs, data.ring_ground_truth,
                                db_path=config.WATCHLIST_DP_DB_PATH, label="E")
        metrics["setup_E"].update(ring_dp)

    with stage("EVALUATE: metrics.json + plots", "evaluate"):
        metrics["timings_seconds"] = TIMINGS
        save_metrics(metrics)
        make_all_plots(metrics)
        print(f"  Saved {config.METRICS_PATH} and plots in {config.RESULTS_DIR}")

    with stage("EXPORT: ONNX (Setup B model)", "export"):
        metrics["onnx"] = export_onnx.run(fed_model, data.n_features, sample=data.X_test[0])

    TIMINGS["total"] = round(time.time() - t_start, 1)
    metrics["timings_seconds"] = TIMINGS
    save_metrics(metrics)

    print("\n=== FINAL COMPARISON ===")
    print(summary_table(metrics))
    print("\nTimings (s): " + ", ".join(f"{k}={v}" for k, v in TIMINGS.items()))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full Federated ScamShield pipeline.")
    parser.add_argument("--data", type=Path, default=config.DATA_FILE, help="PaySim CSV path")
    main(parser.parse_args().data)
