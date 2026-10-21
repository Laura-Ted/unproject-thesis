"""Deciding which prediction is which annotated person.

Two matchers, because ADR-0004 §5.6 asks for both and they are not the same
algorithm. They are named by what they do, never by whose they are (ADR-0005
§3): reproducing the released evaluator's association does not make the whole
metric official, and the config value that selects each one says so.

`mupots_greedy_2d_40px` reproduces what the released MATLAB evaluator does: for
each ground-truth person, the prediction sharing the most joints within 40
pixels on each axis. Greedy, first-come, and a fixed pixel radius that does not
scale with resolution - which matters here, because MuPoTS is 2048x2048 for
five sequences and 1920x1080 for the rest, so the same 40 px is a different
fraction of the body in each. It is reproduced rather than improved because the
association has to stay comparable with published ones.

`normalized_global_assignment` is minimum-cost bipartite assignment on distance
normalised by the image diagonal, with an explicit gate. It is the better
algorithm - greedy matching can take a prediction that a later, closer person
needed - and it is what the project's own diagnostics use.

Neither ever looks at 3D. A match chosen by 3D proximity would make the 3D
metrics that follow partly circular: the pairing would already encode the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unproject.ground_truth.config import MUPOTS_GREEDY_RADIUS_PX
from unproject.ground_truth.dataset import BoolArray, FloatArray, GroundTruthError


@dataclass(frozen=True, slots=True)
class Assignment:
    """Who was paired with whom, and who was left over.

    The leftovers are the point: ADR-0004 §5.6 forbids dropping either side
    silently, and they are what `artifacts.Coverage` counts (§6.1).
    """

    # (ground-truth index, prediction index), in ground-truth order.
    pairs: tuple[tuple[int, int], ...]
    unmatched_ground_truth: tuple[int, ...]
    unmatched_predictions: tuple[int, ...]
    # The cost each pair was accepted at, in the matcher's own units: shared
    # joints for the greedy matcher, normalised distance for the global one.
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.scores) != len(self.pairs):
            raise GroundTruthError(f"{len(self.scores)} scores for {len(self.pairs)} pairs")


def _pairwise_axis_hits(
    gt_2d: FloatArray, gt_valid: BoolArray, pred_2d: FloatArray, pred_valid: BoolArray
) -> tuple[FloatArray, FloatArray]:
    """Joints agreeing within the greedy matcher's radius, and joints visible in both.

    Returns (G, P) counts. The greedy matcher ranks by hits alone; a pair with
    no commonly visible joint has none and is never matched.
    """
    both = gt_valid[:, None, :] & pred_valid[None, :, :]
    delta = np.abs(gt_2d[:, None, :, :] - pred_2d[None, :, :, :])
    within = (delta[..., 0] <= MUPOTS_GREEDY_RADIUS_PX) & (delta[..., 1] <= MUPOTS_GREEDY_RADIUS_PX)
    return (within & both).sum(axis=2).astype(np.float64), both.sum(axis=2).astype(np.float64)


def match_mupots_greedy(
    gt_2d: FloatArray, gt_valid: BoolArray, pred_2d: FloatArray, pred_valid: BoolArray
) -> Assignment:
    """The released evaluator's greedy association, reproduced.

    Each ground-truth person in source order takes the unclaimed prediction with
    the most joints inside the radius. A tie goes to the lower prediction index,
    which is the stable source order ADR-0004 §5.6 requires and is also what
    MATLAB's `max` returns.

    A person with no joint inside the radius of any prediction is unmatched. The
    original does not gate on anything else, and neither does this.
    """
    _check_shapes(gt_2d, gt_valid, pred_2d, pred_valid)
    hits, _ = _pairwise_axis_hits(gt_2d, gt_valid, pred_2d, pred_valid)

    taken: set[int] = set()
    pairs: list[tuple[int, int]] = []
    scores: list[float] = []
    for gt_index in range(gt_2d.shape[0]):
        best = -1
        best_hits = 0.0
        for pred_index in range(pred_2d.shape[0]):
            if pred_index in taken:
                continue
            if hits[gt_index, pred_index] > best_hits:
                best, best_hits = pred_index, float(hits[gt_index, pred_index])
        if best < 0:
            continue
        taken.add(best)
        pairs.append((gt_index, best))
        scores.append(best_hits)

    return _assemble(pairs, scores, gt_2d.shape[0], pred_2d.shape[0])


def match_normalized_global(
    gt_2d: FloatArray,
    gt_valid: BoolArray,
    pred_2d: FloatArray,
    pred_valid: BoolArray,
    *,
    image_diagonal: float,
    gate: float,
) -> Assignment:
    """Minimum-cost bipartite assignment on normalised 2D distance.

    Cost is the mean distance over joints visible in both, divided by the image
    diagonal, so the gate means the same thing at 2048x2048 as at 1920x1080. A
    pair with no commonly visible joint, or a mean distance above the gate, is
    not allowed to match at all.
    """
    from scipy.optimize import linear_sum_assignment

    _check_shapes(gt_2d, gt_valid, pred_2d, pred_valid)
    if image_diagonal <= 0.0:
        raise GroundTruthError(f"image diagonal must be positive, got {image_diagonal}")

    both = gt_valid[:, None, :] & pred_valid[None, :, :]
    distance = np.linalg.norm(gt_2d[:, None, :, :] - pred_2d[None, :, :, :], axis=3)
    shared = both.sum(axis=2)
    with np.errstate(invalid="ignore"):
        cost = np.where(shared > 0, (distance * both).sum(axis=2) / shared, np.inf)
    cost = cost / image_diagonal

    # linear_sum_assignment raises when infinite entries leave no complete
    # assignment, so an ungateable pair is priced above any allowed cost
    # instead and rejected after the fact.
    forbidden = ~np.isfinite(cost) | (cost > gate)
    rows, columns = linear_sum_assignment(np.where(forbidden, gate * 2.0 + 1.0, cost))

    pairs: list[tuple[int, int]] = []
    scores: list[float] = []
    for row, column in zip(rows, columns, strict=True):
        if forbidden[row, column]:
            continue
        pairs.append((int(row), int(column)))
        scores.append(float(cost[row, column]))

    return _assemble(pairs, scores, gt_2d.shape[0], pred_2d.shape[0])


def _assemble(
    pairs: list[tuple[int, int]], scores: list[float], gt_count: int, pred_count: int
) -> Assignment:
    order = sorted(range(len(pairs)), key=lambda index: pairs[index])
    matched_gt = {pair[0] for pair in pairs}
    matched_pred = {pair[1] for pair in pairs}
    return Assignment(
        pairs=tuple(pairs[index] for index in order),
        unmatched_ground_truth=tuple(i for i in range(gt_count) if i not in matched_gt),
        unmatched_predictions=tuple(i for i in range(pred_count) if i not in matched_pred),
        scores=tuple(scores[index] for index in order),
    )


def _check_shapes(
    gt_2d: FloatArray, gt_valid: BoolArray, pred_2d: FloatArray, pred_valid: BoolArray
) -> None:
    """Both sides must already be on the common skeleton.

    Matching arrays of different joint counts by position is the mistake
    ADR-0004 §5.5 exists to prevent; it would pair a wrist with a knee and still
    return a number.
    """
    if gt_2d.shape[1:] != (gt_valid.shape[1], 2) or pred_2d.shape[1:] != (pred_valid.shape[1], 2):
        raise GroundTruthError(
            f"2D joints and validity disagree: {gt_2d.shape}/{gt_valid.shape} against "
            f"{pred_2d.shape}/{pred_valid.shape}"
        )
    if gt_2d.shape[1] != pred_2d.shape[1]:
        raise GroundTruthError(
            f"ground truth has {gt_2d.shape[1]} joints and predictions have "
            f"{pred_2d.shape[1]}. Both sides must be reduced to the common "
            "skeleton before matching; matching by position would pair whatever "
            "happens to sit at the same index."
        )
    if not np.isfinite(gt_2d[gt_valid]).all() or not np.isfinite(pred_2d[pred_valid]).all():
        raise GroundTruthError("a joint marked valid is not finite")
