"""Quality metrics: WER, exact match accuracy."""

import logging
from dataclasses import dataclass

from jiwer import wer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityResult:
    wer_score: float
    exact_match_accuracy: float
    num_samples: int


def compute_quality(
    predictions: list[str],
    references: list[str],
) -> QualityResult:
    """Compute WER and exact-match accuracy.

    Args:
        predictions: Model-generated text outputs.
        references: Ground truth answers/captions.

    Returns:
        QualityResult with WER and accuracy.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    n = len(predictions)
    if n == 0:
        return QualityResult(wer_score=0.0, exact_match_accuracy=0.0, num_samples=0)

    # Normalize for exact match
    preds_norm = [p.strip().lower() for p in predictions]
    refs_norm = [r.strip().lower() for r in references]

    exact_matches = sum(1 for p, r in zip(preds_norm, refs_norm) if p == r)
    accuracy = exact_matches / n

    # WER (filter empty references)
    valid_pairs = [
        (p, r) for p, r in zip(predictions, references) if r.strip()
    ]
    if valid_pairs:
        preds_for_wer, refs_for_wer = zip(*valid_pairs)
        wer_score = wer(list(refs_for_wer), list(preds_for_wer))
    else:
        wer_score = 0.0

    return QualityResult(
        wer_score=wer_score,
        exact_match_accuracy=accuracy,
        num_samples=n,
    )
