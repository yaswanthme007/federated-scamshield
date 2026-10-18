"""Metrics, metrics.json writer and result plots."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config  # noqa: E402

# Validated categorical slots (blue, orange, aqua, yellow) + neutral inks.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def recall_at_fpr(y_true, y_score, target_fpr=config.TARGET_FPR):
    """Highest recall (TPR) achievable while keeping FPR <= ``target_fpr``."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def best_f1(y_true, y_score):
    """F1-optimal threshold from the precision-recall curve.

    Returns:
        (f1, threshold, precision, recall) at the best operating point.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    p, r = precision[:-1], recall[:-1]     # last point has no threshold
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return float(f1[i]), float(thresholds[i]), float(p[i]), float(r[i])


def compute_metrics(y_true, y_score):
    """All headline metrics for one model's test-set scores.

    Note: the F1 threshold is chosen on the test set itself, so F1 is an
    optimistic (oracle-threshold) figure; PR-AUC / ROC-AUC are threshold-free.
    """
    y_true = np.asarray(y_true).astype(int)
    f1, thr, prec, rec = best_f1(y_true, y_score)
    cm = confusion_matrix(y_true, (y_score >= thr).astype(int), labels=[0, 1])
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, y_score),
        "f1": f1,
        "precision_at_best_f1": prec,
        "recall_at_best_f1": rec,
        "best_threshold": thr,
        "confusion_matrix": cm.tolist(),
    }


def format_metrics(m):
    """One-line summary string for console logs."""
    return (f"PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
            f"Recall@1%FPR={m['recall_at_1pct_fpr']:.4f}  F1={m['f1']:.4f}")


def save_metrics(metrics, path=config.METRICS_PATH):
    """Write the metrics dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def _style_axes(ax):
    """Recessive grid/axes, neutral ink."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_pr_auc_comparison(metrics, path=config.PLOT_PR_AUC_PATH):
    """Bar chart of PR-AUC for Setups A (avg), B, C and E (if present)."""
    labels, values = [], []
    names = {"setup_A": "A · Local (avg)", "setup_B": "B · Federated",
             "setup_C": "C · Centralised", "setup_E": "E · Federated + DP"}
    for key, name in names.items():
        if key in metrics and "pr_auc" in metrics[key]:
            labels.append(name)
            values.append(metrics[key]["pr_auc"])
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _style_axes(ax)
    bars = ax.bar(labels, values, color=SERIES_COLORS[0], width=0.55,
                  edgecolor=SURFACE, linewidth=2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, color=INK_PRIMARY)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("PR-AUC (test)", color=INK_SECONDARY)
    ax.set_title("PR-AUC by setup", color=INK_PRIMARY, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fed_convergence(rounds_b, rounds_e=None, central_pr_auc=None,
                         path=config.PLOT_CONVERGENCE_PATH):
    """Line chart of global-model PR-AUC per federated round (B, optionally E)."""
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    _style_axes(ax)
    series = [("B · FedAvg", rounds_b, SERIES_COLORS[0])]
    if rounds_e:
        series.append(("E · FedAvg + DP", rounds_e, SERIES_COLORS[1]))
    for name, rounds, color in series:
        x = [r["round"] for r in rounds]
        y = [r["pr_auc"] for r in rounds]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7, label=name)
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9, color=INK_PRIMARY)
    if central_pr_auc is not None:
        ax.axhline(central_pr_auc, color=INK_SECONDARY, linestyle="--", linewidth=1)
        ax.text(1, central_pr_auc, f"  C · centralised ceiling {central_pr_auc:.3f}",
                va="bottom", fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("Federated round", color=INK_SECONDARY)
    ax.set_ylabel("PR-AUC (test)", color=INK_SECONDARY)
    ax.set_xticks([r["round"] for r in rounds_b])
    ax.set_title("Federated convergence", color=INK_PRIMARY, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, title="Setup B · confusion matrix", path=config.PLOT_CONFUSION_B_PATH):
    """Heatmap of a 2x2 confusion matrix (single-hue, log-scaled shading)."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor=SURFACE)
    ax.imshow(np.log1p(cm), cmap="Blues")
    shade_cut = np.log1p(cm).max() * 0.6
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                color="white" if np.log1p(v) > shade_cut else INK_PRIMARY)
    ax.set_xticks([0, 1], ["Legit", "Fraud"])
    ax.set_yticks([0, 1], ["Legit", "Fraud"])
    ax.set_xlabel("Predicted", color=INK_SECONDARY)
    ax.set_ylabel("Actual", color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=INK_PRIMARY, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all_plots(metrics):
    """Render every results/*.png from a metrics dict."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_pr_auc_comparison(metrics)
    if "setup_B" in metrics:
        plot_fed_convergence(
            metrics["setup_B"]["per_round"],
            metrics.get("setup_E", {}).get("per_round"),
            metrics.get("setup_C", {}).get("pr_auc"),
        )
        plot_confusion_matrix(metrics["setup_B"]["confusion_matrix"])


if __name__ == "__main__":
    with open(config.METRICS_PATH) as f:
        make_all_plots(json.load(f))
    print(f"  Plots written to {config.RESULTS_DIR}")
