"""Tests for person association.

Two matchers with deliberately different behaviour, so most cases are run
against both and the ones that separate them are named after what separates
them.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from unproject.ground_truth.config import MUPOTS_GREEDY_RADIUS_PX
from unproject.ground_truth.dataset import GroundTruthError
from unproject.ground_truth.matching import (
    Assignment,
    match_mupots_greedy,
    match_normalized_global,
)

JOINTS = 5
DIAGONAL = float(np.hypot(1920, 1080))

type Matcher = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Assignment]


def _native(
    gt: np.ndarray, gt_valid: np.ndarray, pred: np.ndarray, pred_valid: np.ndarray
) -> Assignment:
    return match_normalized_global(
        gt, gt_valid, pred, pred_valid, image_diagonal=DIAGONAL, gate=0.1
    )


BOTH: list[tuple[str, Matcher]] = [
    ("mupots_greedy_2d_40px", match_mupots_greedy),
    ("normalized_global_assignment", _native),
]


def _person(x: float, y: float = 500.0) -> np.ndarray:
    """Five joints in a small vertical column at (x, y)."""
    return np.stack([np.full(JOINTS, x), y + np.arange(JOINTS) * 20.0], axis=1)


def _valid(count: int, joints: int = JOINTS) -> np.ndarray:
    return np.ones((count, joints), dtype=np.bool_)


@pytest.mark.parametrize(("name", "matcher"), BOTH, ids=[n for n, _ in BOTH])
class TestBothMatchers:
    def test_an_empty_frame_matches_nothing_and_loses_nobody(
        self, name: str, matcher: Matcher
    ) -> None:
        gt = np.stack([_person(300.0), _person(900.0)])

        result = matcher(gt, _valid(2), np.zeros((0, JOINTS, 2)), _valid(0))

        assert result.pairs == ()
        assert result.unmatched_ground_truth == (0, 1)
        assert result.unmatched_predictions == ()

    def test_nobody_annotated_makes_every_prediction_unmatched(
        self, name: str, matcher: Matcher
    ) -> None:
        pred = np.stack([_person(300.0)])

        result = matcher(np.zeros((0, JOINTS, 2)), _valid(0), pred, _valid(1))

        assert result.unmatched_predictions == (0,)

    def test_source_order_does_not_change_who_pairs_with_whom(
        self, name: str, matcher: Matcher
    ) -> None:
        """§8.1: the same two people, presented both ways round. An assignment
        that depended on order would make the metrics depend on it too."""
        left, right = _person(300.0), _person(900.0)
        gt = np.stack([left, right])
        forward = matcher(gt, _valid(2), np.stack([left, right]), _valid(2))
        reversed_ = matcher(gt, _valid(2), np.stack([right, left]), _valid(2))

        assert forward.pairs == ((0, 0), (1, 1))
        assert reversed_.pairs == ((0, 1), (1, 0))

    def test_one_missed_and_one_spurious(self, name: str, matcher: Matcher) -> None:
        """§8.1: neither side may be dropped silently."""
        gt = np.stack([_person(300.0), _person(900.0)])
        pred = np.stack([_person(300.0), _person(1500.0)])

        result = matcher(gt, _valid(2), pred, _valid(2))

        assert result.pairs == ((0, 0),)
        assert result.unmatched_ground_truth == (1,)
        assert result.unmatched_predictions == (1,)

    def test_only_commonly_visible_joints_are_compared(self, name: str, matcher: Matcher) -> None:
        """A joint the release did not annotate cannot make a pair look worse."""
        gt = np.stack([_person(300.0)])
        pred = gt.copy()
        pred[0, 3:] += 5000.0
        gt_valid, pred_valid = _valid(1), _valid(1)
        gt_valid[0, 3:] = False

        assert matcher(gt, gt_valid, pred, pred_valid).pairs == ((0, 0),)

    def test_a_pair_with_no_shared_joint_cannot_match(self, name: str, matcher: Matcher) -> None:
        gt = np.stack([_person(300.0)])
        gt_valid, pred_valid = _valid(1), _valid(1)
        gt_valid[0, :3] = False
        pred_valid[0, 3:] = False

        result = matcher(gt, gt_valid, gt.copy(), pred_valid)

        assert result.pairs == ()
        assert result.unmatched_ground_truth == (0,)

    def test_a_different_joint_count_is_refused(self, name: str, matcher: Matcher) -> None:
        gt = np.stack([_person(300.0)])
        pred = np.zeros((1, JOINTS - 1, 2))

        with pytest.raises(GroundTruthError, match="common skeleton"):
            matcher(gt, _valid(1), pred, _valid(1, JOINTS - 1))

    def test_a_valid_joint_that_is_not_finite_is_refused(self, name: str, matcher: Matcher) -> None:
        gt = np.stack([_person(300.0)])
        gt[0, 0, 0] = np.nan

        with pytest.raises(GroundTruthError, match="not finite"):
            matcher(gt, _valid(1), gt.copy(), _valid(1))


class TestTheOfficialMatcher:
    def test_the_radius_is_per_axis_not_a_distance(self) -> None:
        """39 px on each axis is 55 px away, and the released evaluator counts
        it as a hit. Reading the threshold as a radius would not."""
        gt = np.stack([_person(300.0)])
        pred = gt + np.array([MUPOTS_GREEDY_RADIUS_PX - 1.0, MUPOTS_GREEDY_RADIUS_PX - 1.0])

        assert match_mupots_greedy(gt, _valid(1), pred, _valid(1)).pairs == ((0, 0),)

    def test_beyond_the_radius_on_one_axis_is_enough_to_miss(self) -> None:
        gt = np.stack([_person(300.0)])
        pred = gt + np.array([MUPOTS_GREEDY_RADIUS_PX + 1.0, 0.0])

        assert match_mupots_greedy(gt, _valid(1), pred, _valid(1)).pairs == ()

    def test_it_ranks_by_how_many_joints_agree(self) -> None:
        gt = np.stack([_person(300.0)])
        close, partial = gt[0].copy(), gt[0].copy()
        partial[2:] += 500.0

        result = match_mupots_greedy(gt, _valid(1), np.stack([partial, close]), _valid(2))

        assert result.pairs == ((0, 1),)
        assert result.scores == (float(JOINTS),)

    def test_a_tie_goes_to_the_lower_index(self) -> None:
        """Stable source order, which is also what MATLAB's `max` returns."""
        gt = np.stack([_person(300.0)])
        pred = np.stack([_person(300.0), _person(300.0)])

        assert match_mupots_greedy(gt, _valid(1), pred, _valid(2)).pairs == ((0, 0),)

    def test_greedy_can_take_the_person_a_later_one_needed(self) -> None:
        """The known weakness of the released algorithm, pinned rather than
        fixed: the headline number has to stay comparable with published ones.

        Person A is a little away from prediction 0 and person B is on top of it.
        A goes first, takes it, and B is left with nothing in range.
        """
        gt = np.stack([_person(300.0 + MUPOTS_GREEDY_RADIUS_PX - 1.0), _person(300.0)])
        pred = np.stack([_person(300.0)])

        result = match_mupots_greedy(gt, _valid(2), pred, _valid(1))

        assert result.pairs == ((0, 0),)
        assert result.unmatched_ground_truth == (1,)


class TestTheNativeMatcher:
    def test_it_does_not_make_the_greedy_mistake(self) -> None:
        """The same case the official matcher gets wrong. Minimum total cost
        gives the prediction to the person who is actually on it."""
        gt = np.stack([_person(300.0 + MUPOTS_GREEDY_RADIUS_PX - 1.0), _person(300.0)])
        pred = np.stack([_person(300.0)])

        result = _native(gt, _valid(2), pred, _valid(1))

        assert result.pairs == ((1, 0),)
        assert result.unmatched_ground_truth == (0,)

    def test_the_gate_is_a_fraction_of_the_image_so_resolution_does_not_move_it(
        self,
    ) -> None:
        """The official 40 px is a smaller share of a 2048-px frame than of a
        1920-px one. This gate is not."""
        gt = np.stack([_person(300.0)])
        pred = gt + np.array([100.0, 0.0])
        wide = float(np.hypot(1920, 1080))
        square = float(np.hypot(2048, 2048))

        gate = 100.0 / wide
        assert match_normalized_global(
            gt, _valid(1), pred, _valid(1), image_diagonal=wide, gate=gate * 1.01
        ).pairs == ((0, 0),)
        assert match_normalized_global(
            gt, _valid(1), pred, _valid(1), image_diagonal=square, gate=gate * 1.01
        ).pairs == ((0, 0),)

    def test_beyond_the_gate_is_not_matched_at_any_cost(self) -> None:
        """Bipartite assignment will pair everything it can unless something
        stops it; a lone prediction and a lone person on opposite sides of the
        frame are not the same person."""
        gt = np.stack([_person(100.0)])
        pred = np.stack([_person(1800.0)])

        result = _native(gt, _valid(1), pred, _valid(1))

        assert result.pairs == ()
        assert result.unmatched_ground_truth == (0,)
        assert result.unmatched_predictions == (0,)

    def test_a_non_positive_diagonal_is_refused(self) -> None:
        gt = np.stack([_person(300.0)])

        with pytest.raises(GroundTruthError, match="diagonal"):
            match_normalized_global(
                gt, _valid(1), gt.copy(), _valid(1), image_diagonal=0.0, gate=0.1
            )
