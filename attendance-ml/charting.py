"""
Charting helpers for the Employee Monitoring System.

Pure presentation: takes a results dict from train_with_real_data.py
and writes PNG charts to ./charts/. No ML, no training, no JSON I/O.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


TASK_TITLES = {
    "employee_activity":   "Employee Activity\nRecognition",
    "office_occupancy":    "Office Occupancy\nDetection",
    "employee_performance":"Employee Performance\nClassification",
}

TASK_COLORS = {
    "employee_activity":   "#3498db",
    "office_occupancy":    "#2ecc71",
    "employee_performance":"#e74c3c",
}

TASK_YLIMS = {
    "employee_activity":   (80, 100),
    "office_occupancy":    (80, 100),
    "employee_performance":(40, 100),
}


def _title(task_key: str) -> str:
    return TASK_TITLES.get(task_key, task_key)


def _title_oneline(task_key: str) -> str:
    return _title(task_key).replace("\n", " ")


def plot_accuracy_bars(results: dict, path: str = None) -> str:
    if path is None:
        path = os.path.join(OUTPUT_DIR, "chart1_accuracy_comparison.png")
    print("  [1/3] Accuracy comparison...")

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))
    if len(results) == 1:
        axes = [axes]

    for i, (task_key, data) in enumerate(results.items()):
        names = list(data["scores"].keys())
        vals  = [v * 100 for v in data["scores"].values()]
        best  = data["best_accuracy"] * 100
        color = TASK_COLORS.get(task_key, "#3498db")
        ylim  = TASK_YLIMS.get(task_key, (40, 100))

        bar_colors = ["#2ecc71" if v == max(vals) else color for v in vals]
        bars = axes[i].bar(range(len(names)), vals, color=bar_colors,
                           edgecolor="black", linewidth=0.5)
        axes[i].set_xticks(range(len(names)))
        axes[i].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        axes[i].set_title(_title(task_key), fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Accuracy (%)")
        axes[i].set_ylim(ylim)
        axes[i].axhline(y=best, color="red", linestyle="--", alpha=0.5,
                        label=f"Best:{best:.1f}%")
        axes[i].legend(fontsize=8)

        for b, v in zip(bars, vals):
            axes[i].text(b.get_x() + b.get_width() / 2., b.get_height() + 0.3,
                         f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    SAVED: {path}")
    return path


def plot_confusion_matrices(results: dict, path: str = None) -> str:
    if path is None:
        path = os.path.join(OUTPUT_DIR, "chart2_confusion_matrices.png")

    items = [(k, d) for k, d in results.items() if "confusion_matrix" in d]
    if not items:
        return None

    print("  [2/3] Confusion matrices...")
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 4.5))
    if len(items) == 1:
        axes = [axes]
    cmaps = ["Blues", "Greens", "Reds"]

    for i, (task_key, data) in enumerate(items):
        cm = np.array(data["confusion_matrix"])
        classes = data.get("classes", [str(j) for j in range(cm.shape[0])])
        best = data["best_model"]
        acc = data["best_accuracy"] * 100

        axes[i].imshow(cm, cmap=cmaps[i % 3], interpolation="nearest")
        axes[i].set_title(
            f"{_title(task_key)}\n({best},{acc:.1f}%)",
            fontsize=9, fontweight="bold")
        short = [c[:10] for c in classes]
        axes[i].set_xticks(range(len(short)))
        axes[i].set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        axes[i].set_yticks(range(len(short)))
        axes[i].set_yticklabels(short, fontsize=7)
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")

        threshold = cm.max() / 2.0
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                axes[i].text(c, r, str(cm[r, c]),
                             ha="center", va="center", fontsize=8,
                             fontweight="bold",
                             color="white" if cm[r, c] > threshold else "black")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    SAVED: {path}")
    return path


def plot_best_models_summary(results: dict, path: str = None) -> str:
    if path is None:
        path = os.path.join(OUTPUT_DIR, "chart3_best_models.png")
    print("  [3/3] Summary chart...")

    labels, accs, colors = [], [], []
    for i, (task_key, data) in enumerate(results.items()):
        labels.append(f"{data['best_model']}\n({_title_oneline(task_key)})")
        accs.append(data["best_accuracy"] * 100)
        colors.append(TASK_COLORS.get(task_key, "#3498db"))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(labels)), accs,
                   color=colors, edgecolor="black", height=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Best Model Per Task — Employee Monitoring System",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 110)

    for b, v in zip(bars, accs):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2.,
                f"{v:.2f}%", va="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    SAVED: {path}")
    return path


def plot_all(results: dict) -> list:
    paths = []
    paths.append(plot_accuracy_bars(results))
    cm = plot_confusion_matrices(results)
    if cm:
        paths.append(cm)
    paths.append(plot_best_models_summary(results))
    return paths
