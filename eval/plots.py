"""Charts for the evaluation: confusion matrices with the expensive cells called out, and the auto-reply precision/recall curve with the operating point marked.

Every chart is written to eval/results/ as a PNG by run_eval.py, so the figures in
the presentation and the numbers in the repository come from the same run. Nothing here computes a metric — it only draws what run_eval.py already calculated.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # No display in a terminal session; write straight to file.
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

# Cells that cost money or trust, as (true, predicted) pairs. A confusion matrix
# treats every off-diagonal cell alike; a bank does not.
EXPENSIVE_CELLS: set[tuple[str, str]] = {
    ("human", "auto_reply"),  # answered automatically when a person was needed
    ("human", "action"),  # proposed an action when a person was needed
    ("security_fraud", "cards_issuing"),
    ("security_fraud", "payments_transfers"),
    ("security_fraud", "account_servicing"),
    ("security_fraud", "topup_balance"),
    ("security_fraud", "fx_international"),
}


def plot_confusion(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    """Heatmap with counts written in, and expensive cells outlined in red."""
    figure, axes = plt.subplots(figsize=(1.4 * len(matrix) + 3, 1.2 * len(matrix) + 2.5))
    axes.imshow(matrix.to_numpy(), cmap="Blues")

    for row, true_label in enumerate(matrix.index):
        for column, predicted_label in enumerate(matrix.columns):
            count = int(matrix.iloc[row, column])
            expensive = (true_label, predicted_label) in EXPENSIVE_CELLS and count > 0
            axes.text(
                column,
                row,
                str(count),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold" if expensive else "normal",
                color="#b34a3c" if expensive else ("white" if count > matrix.to_numpy().max() / 2 else "black"),
            )
            if expensive:
                axes.add_patch(
                    plt.Rectangle(
                        (column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#b34a3c", linewidth=2.5
                    )
                )

    axes.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    axes.set_yticks(range(len(matrix.index)), matrix.index)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_title(f"{title}\nred = expensive errors")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_precision_recall(sweep: pd.DataFrame, chosen_threshold: float, output_path: Path) -> None:
    """Auto-reply precision and recall against the retrieval threshold, operating point marked.

    Two panels rather than a single precision-versus-recall curve: the threshold is
    the thing being chosen, so it belongs on an axis where it can be read off.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))

    left.plot(sweep.threshold, sweep.precision, marker="o", label="auto_reply precision", color="#1f6f4a")
    left.plot(sweep.threshold, sweep.recall, marker="s", label="auto_reply recall", color="#b34a3c")
    left.axvline(chosen_threshold, linestyle="--", color="#444")
    left.set_xlabel("retrieval threshold (top-1 cosine)")
    left.set_ylabel("score")
    left.set_title("Choosing the operating point")
    left.legend()
    left.grid(alpha=0.3)

    at_chosen = sweep.iloc[(sweep.threshold - chosen_threshold).abs().argmin()]
    left.annotate(
        f"chosen {chosen_threshold:.2f}\nP={at_chosen.precision:.2f} R={at_chosen.recall:.2f}",
        xy=(chosen_threshold, at_chosen.precision),
        xytext=(10, -40),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#444"},
    )

    right.plot(sweep.recall, sweep.precision, marker="o", color="#33639e")
    right.scatter([at_chosen.recall], [at_chosen.precision], s=140, facecolor="none",
                  edgecolor="#b34a3c", linewidth=2.5, zorder=5, label=f"chosen ({chosen_threshold:.2f})")
    for _, row in sweep.iterrows():
        right.annotate(f"{row.threshold:.2f}", (row.recall, row.precision), fontsize=7,
                       textcoords="offset points", xytext=(4, 4))
    right.set_xlabel("auto_reply recall (deflection)")
    right.set_ylabel("auto_reply precision")
    right.set_title("Precision / recall trade-off")
    right.set_ylim(-0.05, 1.05)
    right.legend()
    right.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_ablation(table: pd.DataFrame, output_path: Path) -> None:
    """Grouped bars comparing retrieval modes on the same two numbers."""
    figure, axes = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(table))
    axes.bar(positions - 0.2, table.precision, 0.4, label="precision", color="#1f6f4a")
    axes.bar(positions + 0.2, table.recall, 0.4, label="recall", color="#b34a3c")

    for position, row in zip(positions, table.itertuples()):
        axes.text(position - 0.2, row.precision + 0.02, f"{row.precision:.2f}", ha="center", fontsize=9)
        axes.text(position + 0.2, row.recall + 0.02, f"{row.recall:.2f}", ha="center", fontsize=9)

    axes.set_xticks(positions, table["retriever"])
    axes.set_ylabel("score")
    axes.set_ylim(0, 1.15)
    axes.set_title("Retrieval ablation: finding answerable messages\n(rate-matched, so each mode flags the same number)")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
