"""Visualisation for experiment results."""

import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

# Labels match paper Figure 10 exactly.
CONDITION_LABELS: dict[str, str] = {
    "reference": "Reference",
    "teacher": "Teacher",
    "student_aux_same": "Student\n(aux. only)",
    "student_all_same": "Student\n(all logits)",
    "student_aux_diff": "Cross-model\n(aux. only)",
    "student_all_diff": "Cross-model\n(all logits)",
}

CONDITION_COLORS: dict[str, str] = {
    "reference": "#909090",
    "teacher": "#8B5E52",
    "student_aux_same": "#7B52AB",
    "student_all_same": "#7B52AB",
    "student_aux_diff": "#C8A8D8",
    "student_all_diff": "#C8A8D8",
}


def plot_results(
    results: dict[str, list[float]],
    n_classes: int = 10,
    output_path: str | Path = "results.png",
    title: str = "Subliminal learning: MLP experiment",
) -> None:
    """Bar chart of mean test accuracy ± 95% CI for each experimental condition.

    Args:
        n_classes: Used to draw the random-baseline line at ``1 / n_classes``.
        title: Plot title shown above the chart.
    """
    names = [n for n in CONDITION_LABELS if n in results]
    means = [statistics.mean(results[n]) for n in names]
    ci95 = [
        1.96 * statistics.stdev(results[n]) / math.sqrt(len(results[n])) for n in names
    ]
    labels = [CONDITION_LABELS[n] for n in names]

    colors = [CONDITION_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, means, yerr=ci95, capsize=5, color=colors)
    ax.axhline(
        1.0 / n_classes,
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Random baseline",
    )
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
