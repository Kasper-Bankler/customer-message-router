"""Classification metrics and the bootstrap, written out in plain numpy so every number can be explained out loud.

The bootstrap here is the ordinary non-parametric percentile bootstrap, and it is
four lines long on purpose:

    1. draw n row indices with replacement from the n rows we measured
    2. recompute the statistic on that resample
    3. repeat 10,000 times
    4. the 2.5th and 97.5th percentiles of those 10,000 values are the interval

It answers "if I had labelled a different 80 messages from the same process, how
much would this number move?". With n=80 the answer is: quite a lot, and saying
so is the point. Precision, recall and F1 are implemented by hand rather than
imported, because "sklearn computed it" is not an explanation.
"""

from typing import Callable

import numpy as np
import pandas as pd

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile bootstrap interval for any statistic of n paired rows.

    `statistic` takes an array of row indices and returns one number, so the same
    function works for accuracy, macro-F1 or a per-class precision without any of
    them needing to know a bootstrap is happening.
    """
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for draw in range(resamples):
        values[draw] = statistic(rng.integers(0, n, n))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def precision_recall_f1(
    truth: np.ndarray, predicted: np.ndarray, label: str
) -> tuple[float, float, float]:
    """One class, counted the long way. Zero denominators give 0.0, never NaN."""
    true_positive = int(((truth == label) & (predicted == label)).sum())
    false_positive = int(((truth != label) & (predicted == label)).sum())
    false_negative = int(((truth == label) & (predicted != label)).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_f1(truth: np.ndarray, predicted: np.ndarray, labels: list[str]) -> float:
    """Unweighted mean F1 across classes, so a rare class counts as much as a common one.

    This is why it is the headline instead of accuracy: on a gold set that is 78%
    one class, accuracy rewards ignoring the other three.
    """
    return float(np.mean([precision_recall_f1(truth, predicted, label)[2] for label in labels]))


def balanced_accuracy(truth: np.ndarray, predicted: np.ndarray, labels: list[str]) -> float:
    """Mean per-class recall. Its baseline is 1/k whatever the class balance is."""
    recalls = [
        precision_recall_f1(truth, predicted, label)[1] for label in labels if (truth == label).any()
    ]
    return float(np.mean(recalls)) if recalls else 0.0


def majority_baseline(truth: np.ndarray) -> float:
    """Accuracy of a router that ignores the message and always guesses the commonest label."""
    _, counts = np.unique(truth, return_counts=True)
    return float(counts.max() / len(truth))


def f1_statistic(
    truth: np.ndarray, predicted: np.ndarray, label: str
) -> Callable[[np.ndarray], float]:
    """Build the "F1 for this label" statistic the bootstrap resamples.

    A named factory rather than an inline lambda: the returned closure captures
    `label` as a function argument, so it needs no default-argument trick to pin
    the value across loop iterations.
    """

    def statistic(indices: np.ndarray) -> float:
        return precision_recall_f1(truth[indices], predicted[indices], label)[2]

    return statistic


def per_class_table(truth: np.ndarray, predicted: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Precision, recall, F1 and support per class, each with its own bootstrap interval."""
    rows = []
    for label in labels:
        precision, recall, f1 = precision_recall_f1(truth, predicted, label)
        low, high = bootstrap_ci(f1_statistic(truth, predicted, label), len(truth))
        rows.append(
            {
                "class": label,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "f1_ci_low": round(low, 3),
                "f1_ci_high": round(high, 3),
                "support": int((truth == label).sum()),
            }
        )
    return pd.DataFrame(rows)


def confusion_counts(truth: np.ndarray, predicted: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Rows are the true class, columns the predicted one. Counted by hand, no sklearn."""
    matrix = [[int(((truth == t) & (predicted == p)).sum()) for p in labels] for t in labels]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def rule_of_three_upper(n: int) -> float:
    """Upper 95% bound on a failure rate after n trials with zero failures: 3/n.

    The bootstrap cannot help here. Resampling twenty successes only ever produces
    successes, so it reports [1.000, 1.000] — an interval that looks like certainty
    and is really just zero variance. "20/20, and with this much evidence the true
    failure rate could still be as high as 15%" is the honest sentence.
    """
    return 3.0 / n if n else 1.0


def is_degenerate(values: np.ndarray) -> bool:
    """True when every observation is identical, so any bootstrap interval is a point."""
    return len(values) > 0 and bool(np.all(values == values[0]))


def with_ci(name: str, value: float, statistic: Callable[[np.ndarray], float], n: int) -> dict:
    """Package one headline number with its interval and the n it rests on."""
    low, high = bootstrap_ci(statistic, n)
    return {name: round(value, 4), f"{name}_ci95": [round(low, 4), round(high, 4)], f"{name}_n": n}
