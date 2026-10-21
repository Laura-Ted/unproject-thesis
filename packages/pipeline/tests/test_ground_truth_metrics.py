"""Tests for the error measures, on hand-built cases.

Every number here was decided before the code ran: a joint placed exactly
149 mm away, a root offset chosen so absolute error must move by a known amount
and root-relative error must not move at all. ADR-0004 §8.1 lists the cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from unproject.ground_truth.dataset import GroundTruthError
from unproject.ground_truth.metrics import (
    SCALE_PROXY_SEGMENTS,
    aggregate,
    depth_bins,
    pck,
    per_joint,
    person_errors,
    root_error,
    scale_proxy,
)

ROOT = 0
JOINTS = 4


def _pose(depth_m: float = 3.0) -> np.ndarray:
    """Four joints: a root at the given depth and three hanging off it."""
    return np.array(
        [
            [0.0, 0.0, depth_m],
            [0.2, -0.4, depth_m],
            [-0.2, -0.4, depth_m],
            [0.0, 0.5, depth_m],
        ]
    )


def _all_valid() -> np.ndarray:
    return np.ones(JOINTS, dtype=np.bool_)


class TestOnePerson:
    def test_a_perfect_prediction_scores_zero(self) -> None:
        pose = _pose()

        errors = person_errors(pose, pose.copy(), _all_valid(), root_index=ROOT)

        assert errors.absolute_mm == pytest.approx(np.zeros(JOINTS))
        assert errors.root_relative_mm == pytest.approx(np.zeros(JOINTS))
        assert errors.root_error_xyz_mm == pytest.approx(np.zeros(3))

    def test_a_root_offset_moves_absolute_error_and_not_relative(self) -> None:
        """The distinction the whole module is built around. A pose that is
        exactly right but placed 30 cm too far away is a depth failure, not an
        articulation failure, and the two metrics must say so separately."""
        gt = _pose()
        shifted = gt + np.array([0.0, 0.0, 0.3])

        errors = person_errors(gt, shifted, _all_valid(), root_index=ROOT)

        assert errors.absolute_mm == pytest.approx(np.full(JOINTS, 300.0))
        assert errors.root_relative_mm == pytest.approx(np.zeros(JOINTS))
        assert errors.root_error_xyz_mm == pytest.approx([0.0, 0.0, 300.0])

    def test_the_signed_root_error_keeps_its_direction(self) -> None:
        """Too near and too far must not average away to nothing."""
        gt = _pose()

        near = person_errors(gt, gt - np.array([0.0, 0.0, 0.3]), _all_valid(), root_index=ROOT)
        far = person_errors(gt, gt + np.array([0.0, 0.0, 0.3]), _all_valid(), root_index=ROOT)

        assert near.root_error_xyz_mm[2] == pytest.approx(-300.0)
        assert far.root_error_xyz_mm[2] == pytest.approx(300.0)

    def test_metres_become_millimetres_exactly_once(self) -> None:
        """A second conversion would report 300 metres or 0.3 mm; both look
        wrong only if you know what to expect."""
        gt = _pose()

        errors = person_errors(gt, gt + np.array([0.001, 0.0, 0.0]), _all_valid(), root_index=ROOT)

        assert errors.absolute_mm[1] == pytest.approx(1.0)

    def test_an_invalid_joint_is_excluded_rather_than_zeroed(self) -> None:
        gt = _pose()
        valid = _all_valid()
        valid[2] = False

        errors = person_errors(gt, gt + 0.1, valid, root_index=ROOT)

        assert np.isnan(errors.absolute_mm[2])
        assert not np.isnan(errors.absolute_mm[1])

    def test_a_missing_root_is_refused(self) -> None:
        valid = _all_valid()
        valid[ROOT] = False

        with pytest.raises(GroundTruthError, match="nothing to align on"):
            person_errors(_pose(), _pose(), valid, root_index=ROOT)

    def test_mismatched_shapes_are_refused(self) -> None:
        with pytest.raises(GroundTruthError, match="matching"):
            person_errors(_pose(), _pose()[:3], _all_valid(), root_index=ROOT)


class TestThePckBoundary:
    """ADR-0004 §8.1 asks for 149 and 151 explicitly rather than trusting the
    reading of `<=`."""

    def _table(self, *errors: float) -> tuple[np.ndarray, np.ndarray]:
        table = np.array([list(errors)])
        return table, np.ones_like(table, dtype=np.bool_)

    @pytest.mark.parametrize(
        ("error_mm", "expected"), [(149.0, 100.0), (150.0, 100.0), (151.0, 0.0)]
    )
    def test_the_threshold_is_inclusive(self, error_mm: float, expected: float) -> None:
        errors, valid = self._table(error_mm)

        assert pck(errors, valid, radius_mm=150.0).micro == pytest.approx(expected)

    def test_half_in_half_out(self) -> None:
        errors, valid = self._table(149.0, 151.0)

        assert pck(errors, valid, radius_mm=150.0).micro == pytest.approx(50.0)

    def test_a_non_positive_radius_is_refused(self) -> None:
        errors, valid = self._table(1.0)

        with pytest.raises(GroundTruthError, match="must be positive"):
            pck(errors, valid, radius_mm=0.0)


class TestAggregation:
    def test_micro_and_macro_differ_when_people_contribute_unevenly(self) -> None:
        """One person with four good joints and one with a single bad one.
        Pooled, the good person dominates; per person, they weigh the same. A
        report that carried only one of these would hide the other."""
        errors = np.array([[10.0, 10.0, 10.0, 10.0], [200.0, np.nan, np.nan, np.nan]])
        valid = np.array([[True] * 4, [True, False, False, False]])

        result = aggregate(errors, valid)

        assert result.micro == pytest.approx((10.0 * 4 + 200.0) / 5)
        assert result.macro == pytest.approx((10.0 + 200.0) / 2)
        assert (result.people, result.joints) == (2, 5)

    def test_a_person_with_no_valid_joint_does_not_count_as_perfect(self) -> None:
        errors = np.array([[10.0, 10.0], [np.nan, np.nan]])
        valid = np.array([[True, True], [False, False]])

        result = aggregate(errors, valid)

        assert result.people == 1
        assert result.macro == pytest.approx(10.0)

    def test_nothing_valid_is_empty_rather_than_zero(self) -> None:
        errors = np.full((2, 2), np.nan)
        valid = np.zeros((2, 2), dtype=np.bool_)

        result = aggregate(errors, valid)

        assert result.joints == 0
        assert np.isnan(result.micro)

    def test_an_error_for_a_valid_joint_cannot_be_missing(self) -> None:
        with pytest.raises(GroundTruthError, match="cannot be both"):
            aggregate(np.array([[np.nan]]), np.array([[True]]))


class TestPerJoint:
    def test_a_joint_nobody_annotated_is_flagged_not_zeroed(self) -> None:
        """Zero is a perfect score. An absent joint is not one."""
        errors = np.array([[10.0, np.nan], [30.0, np.nan]])
        valid = np.array([[True, False], [True, False]])

        means, present = per_joint(errors, valid)

        assert means[0] == pytest.approx(20.0)
        assert np.isnan(means[1])
        assert present.tolist() == [True, False]


class TestRootError:
    def test_bias_and_magnitude_are_not_the_same_number(self) -> None:
        """Scatter and offset have the same MAE. Only the bias separates a model
        that is wrong in both directions from one that is always too far."""
        scattered = np.array([[0.0, 0.0, 300.0], [0.0, 0.0, -300.0]])
        offset = np.array([[0.0, 0.0, 300.0], [0.0, 0.0, 300.0]])

        scattered_mae, scattered_bias = root_error(scattered)
        offset_mae, offset_bias = root_error(offset)

        assert scattered_mae[2] == pytest.approx(offset_mae[2]) == pytest.approx(300.0)
        assert scattered_bias[2] == pytest.approx(0.0)
        assert offset_bias[2] == pytest.approx(300.0)

    def test_no_people_gives_no_number(self) -> None:
        mae, bias = root_error(np.zeros((0, 3)))

        assert np.isnan(mae).all()
        assert np.isnan(bias).all()


class TestDepthBins:
    def test_edges_produce_one_more_bin_than_edges(self) -> None:
        depths = np.array([1.0, 3.0, 5.0])
        errors = np.zeros((3, 3))

        bins = depth_bins(depths, errors, [2.0, 4.0])

        assert [b.people for b in bins] == [1, 1, 1]

    def test_a_person_exactly_on_an_edge_lands_in_the_upper_bin_only(self) -> None:
        depths = np.array([2.0])
        errors = np.zeros((1, 3))

        bins = depth_bins(depths, errors, [2.0])

        assert [b.people for b in bins] == [0, 1]

    def test_relative_bias_is_the_shape_a_focal_error_takes(self) -> None:
        """A focal length wrong by a constant factor puts everyone too far by a
        constant *proportion*, not a constant distance. Both people here are 20%
        too far, at very different distances."""
        depths = np.array([3.0, 6.0])
        errors = np.array([[0.0, 0.0, 600.0], [0.0, 0.0, 1200.0]])

        [single] = depth_bins(depths, errors, [])

        assert single.relative_bias == pytest.approx(0.2)
        assert single.signed_bias_mm == pytest.approx(900.0)

    def test_an_empty_bin_reports_no_people_rather_than_zero_error(self) -> None:
        bins = depth_bins(np.array([5.0]), np.zeros((1, 3)), [2.0])

        assert bins[0].people == 0
        assert np.isnan(bins[0].absolute_error_mm)

    def test_unsorted_edges_are_refused(self) -> None:
        with pytest.raises(GroundTruthError, match="ascend"):
            depth_bins(np.array([1.0]), np.zeros((1, 3)), [4.0, 2.0])


LEG_NAMES = (
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
)


def _legs(femur_m: float = 0.42, tibia_m: float = 0.40, depth_m: float = 3.0) -> np.ndarray:
    """A pelvis with two straight legs hanging from it, lengths as given."""
    return np.array(
        [
            [0.0, 0.0, depth_m],
            [-0.1, 0.0, depth_m],
            [-0.1, femur_m, depth_m],
            [-0.1, femur_m + tibia_m, depth_m],
            [0.1, 0.0, depth_m],
            [0.1, femur_m, depth_m],
            [0.1, femur_m + tibia_m, depth_m],
        ]
    )


class TestTheScaleProxy:
    """Declared before any analysis reads it; ADR-0005 WP5."""

    def test_it_is_the_bilateral_femur_plus_tibia(self) -> None:
        proxy = scale_proxy(_legs(0.42, 0.40), np.ones(7, dtype=np.bool_), LEG_NAMES)

        assert proxy == pytest.approx(2 * (0.42 + 0.40))
        assert SCALE_PROXY_SEGMENTS == (
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        )

    def test_it_does_not_move_with_where_the_person_is(self) -> None:
        """A length is a length: placing the same body five metres further
        away, or a metre to the left, must not change it. That is what makes it
        usable against the depth error rather than part of it."""
        valid = np.ones(7, dtype=np.bool_)
        here = scale_proxy(_legs(depth_m=2.0), valid, LEG_NAMES)
        there = scale_proxy(_legs(depth_m=7.0) + np.array([1.0, 0.0, 0.0]), valid, LEG_NAMES)

        assert here == pytest.approx(there)

    def test_a_missing_leg_joint_is_nan_not_a_shorter_leg(self) -> None:
        valid = np.ones(7, dtype=np.bool_)
        valid[LEG_NAMES.index("right_knee")] = False

        assert np.isnan(scale_proxy(_legs(), valid, LEG_NAMES))

    def test_a_skeleton_without_legs_is_refused(self) -> None:
        with pytest.raises(GroundTruthError, match="hips, knees and ankles"):
            scale_proxy(_legs()[:3], np.ones(3, dtype=np.bool_), LEG_NAMES[:3])

    def test_person_errors_carries_both_sides(self) -> None:
        gt = _legs(0.42, 0.40)
        # The prediction is a smaller body, 5% shorter in every segment.
        pred = _legs(0.42 * 0.95, 0.40 * 0.95)

        errors = person_errors(gt, pred, np.ones(7, dtype=np.bool_), root_index=0, names=LEG_NAMES)

        assert errors.scale_proxy_annotated_m == pytest.approx(1.64)
        assert errors.scale_proxy_predicted_m == pytest.approx(1.64 * 0.95)

    def test_without_names_the_proxy_is_absent(self) -> None:
        pose = _pose()

        errors = person_errors(pose, pose.copy(), _all_valid(), root_index=ROOT)

        assert np.isnan(errors.scale_proxy_predicted_m)
        assert np.isnan(errors.scale_proxy_annotated_m)
