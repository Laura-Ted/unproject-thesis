"""Tests for the MuPoTS adapter, on a synthetic release.

No real dataset here. The MuPoTS licence forbids redistribution, so ADR-0004
§8.2's fallback applies: a fixture built to the documented schema, with the real
files reserved for local integration runs.

The fixture is a projection of known 3D points through a known camera, so every
assertion has an answer decided in advance rather than read off the output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unproject.ground_truth.dataset import CANONICAL_CONVENTION, GroundTruthError
from unproject.ground_truth.mupots import (
    JOINT_NAMES,
    PELVIS_INDEX,
    SEQUENCE_COUNT,
    MupotsDataset,
    image_size,
    recover_intrinsics,
)

# The camera the fixture is built with; the adapter has to recover it.
FX, FY, CX, CY = 1100.0, 1100.0, 960.0, 540.0


def _skeleton(depth_mm: float, offset_mm: float = 0.0) -> np.ndarray:
    """17 joints in millimetres, spread widely enough to constrain a fit."""
    rng = np.random.default_rng(0)
    joints = rng.uniform(-400.0, 400.0, size=(3, len(JOINT_NAMES)))
    joints[0] += offset_mm
    joints[2] = depth_mm + rng.uniform(-100.0, 100.0, size=len(JOINT_NAMES))
    return joints


def _project(joints_mm: np.ndarray) -> np.ndarray:
    x, y, z = joints_mm
    return np.stack([FX * x / z + CX, FY * y / z + CY])


def _write_release(
    root: Path,
    *,
    sequences: tuple[str, ...] = ("TS9",),
    frames: int = 3,
    slots: int = 2,
    joints: int = len(JOINT_NAMES),
    valid: bool = True,
) -> Path:
    """A MuPoTS-shaped release: TS directories each holding one annot.mat."""
    from scipy.io import savemat

    base = root / "MultiPersonTestSet"
    for sequence in sequences:
        directory = base / sequence
        directory.mkdir(parents=True, exist_ok=True)
        cells = np.empty((frames, slots), dtype=object)
        for frame in range(frames):
            for slot in range(slots):
                three = _skeleton(3000.0 + 500.0 * slot, offset_mm=200.0 * slot)[:, :joints]
                cells[frame, slot] = {
                    "annot2": _project(three),
                    "annot3": three,
                    "univ_annot3": three.copy(),
                    "isValidFrame": np.array([[1 if valid else 0]]),
                }
        savemat(directory / "annot.mat", {"annotations": cells})
    return base


class TestTheSequenceTable:
    def test_all_twenty_identifiers_are_inventoried(self, tmp_path: Path) -> None:
        base = _write_release(tmp_path, sequences=tuple(f"TS{i}" for i in range(1, 21)))
        dataset = MupotsDataset(base, release_identity="fixture")

        assert len(dataset.sequences()) == SEQUENCE_COUNT

    def test_a_partial_release_lists_only_what_is_on_disk(self, tmp_path: Path) -> None:
        """An evaluation over 12 of 20 is not the benchmark, so the table must not
        pretend the rest are there."""
        base = _write_release(tmp_path, sequences=("TS1", "TS2"))
        dataset = MupotsDataset(base, release_identity="fixture")

        assert dataset.sequences() == ["TS1", "TS2"]

    def test_an_empty_directory_says_what_was_expected(self, tmp_path: Path) -> None:
        dataset = MupotsDataset(tmp_path, release_identity="fixture")

        with pytest.raises(GroundTruthError, match="TS1"):
            dataset.sequences()

    def test_the_two_image_sizes(self) -> None:
        # The official table gives [rows cols], and reading it as (width, height)
        # is a swap the recovered optical centre catches immediately.
        assert image_size("TS1") == (2048, 2048)
        assert image_size("TS5") == (2048, 2048)
        assert image_size("TS6") == (1920, 1080)
        assert image_size("TS20") == (1920, 1080)

    @pytest.mark.parametrize("bad", ["TS0", "TS21", "ts1", "TSX", "1"])
    def test_an_identifier_outside_the_release_is_refused(self, bad: str) -> None:
        with pytest.raises(GroundTruthError):
            image_size(bad)


class TestReadingRecords:
    def test_it_normalises_a_frame(self, tmp_path: Path) -> None:
        base = _write_release(tmp_path)
        [record, *_] = MupotsDataset(base, release_identity="fixture").records("TS9")

        assert record.convention == CANONICAL_CONVENTION
        assert (record.image_width, record.image_height) == (1920, 1080)
        assert record.sequence_id == "TS9"
        assert len(record.people) == 2

    def test_millimetres_become_metres(self, tmp_path: Path) -> None:
        """The fixture puts a person near 3 m; in millimetres it would be 3000."""
        base = _write_release(tmp_path)
        [record, *_] = MupotsDataset(base, release_identity="fixture").records("TS9")
        person = record.people[0]
        root = person.root_translation

        assert root is not None
        assert 2.0 < float(root[2]) < 4.0
        assert person.joints_3d.shape == (len(JOINT_NAMES), 3)

    def test_the_root_is_the_pelvis(self, tmp_path: Path) -> None:
        """The official evaluator aligns on joint 15, which is index 14 here."""
        base = _write_release(tmp_path)
        [record, *_] = MupotsDataset(base, release_identity="fixture").records("TS9")
        person = record.people[0]

        assert JOINT_NAMES[PELVIS_INDEX] == "pelvis"
        assert person.root_translation is not None
        assert person.root_translation == pytest.approx(person.joints_3d[PELVIS_INDEX])

    def test_an_absent_person_leaves_the_slot_empty(self, tmp_path: Path) -> None:
        """isValidFrame marks a slot as holding nobody; that is recall, not an error."""
        base = _write_release(tmp_path, valid=False)
        records = MupotsDataset(base, release_identity="fixture").records("TS9")

        assert records
        assert all(record.people == () for record in records)

    def test_a_different_joint_count_fails_closed(self, tmp_path: Path) -> None:
        """A release that changed its joint set must not be silently reshaped."""
        base = _write_release(tmp_path, joints=14)

        with pytest.raises(GroundTruthError, match="14 joint"):
            MupotsDataset(base, release_identity="fixture").records("TS9")

    def test_a_file_that_is_not_an_annotation_is_refused(self, tmp_path: Path) -> None:
        from scipy.io import savemat

        base = tmp_path / "MultiPersonTestSet" / "TS9"
        base.mkdir(parents=True)
        savemat(base / "annot.mat", {"something_else": np.zeros((2, 2))})

        with pytest.raises(GroundTruthError, match="not a MuPoTS annotation file"):
            MupotsDataset(tmp_path / "MultiPersonTestSet", release_identity="f").records("TS9")

    def test_universal_and_subject_scale_are_selectable(self, tmp_path: Path) -> None:
        """The official root-relative protocol wants one, absolute work the other."""
        base = _write_release(tmp_path)

        subject = MupotsDataset(base, release_identity="f").records("TS9")
        universal = MupotsDataset(base, release_identity="f", universal=True).records("TS9")

        assert subject[0].people[0].joints_3d == pytest.approx(universal[0].people[0].joints_3d)


class TestRecoveringTheCamera:
    def test_it_recovers_the_camera_the_fixture_was_built_with(self, tmp_path: Path) -> None:
        base = _write_release(tmp_path, frames=20)

        found = MupotsDataset(base, release_identity="fixture").intrinsics("TS9")

        assert found.camera.fx == pytest.approx(FX, rel=1e-6)
        assert found.camera.fy == pytest.approx(FY, rel=1e-6)
        assert found.camera.cx == pytest.approx(CX, abs=1e-3)
        assert found.camera.cy == pytest.approx(CY, abs=1e-3)
        assert found.residual_px < 1e-6

    def _correspondences(self) -> tuple[np.ndarray, np.ndarray]:
        three = np.concatenate([_skeleton(3000.0 + 100.0 * i) for i in range(8)], axis=1)
        two = np.concatenate([_project(_skeleton(3000.0 + 100.0 * i)) for i in range(8)], axis=1)
        return three.T / 1000.0, two.T

    def test_an_honest_correspondence_fits_exactly(self) -> None:
        three, two = self._correspondences()

        assert recover_intrinsics(three, two).residual_px < 1e-6

    def test_a_flipped_axis_is_caught_by_the_sign_before_the_residual(self) -> None:
        """The vertical convention is checked more sharply than by a residual.

        Reading +Y as up rather than down makes the fitted focal length
        negative, which is not a near miss - it is a statement that the two
        annotations disagree about which way the axis points.
        """
        three, two = self._correspondences()
        flipped = three.copy()
        flipped[:, 1] *= -1.0

        with pytest.raises(GroundTruthError, match="not positive"):
            recover_intrinsics(flipped, two)

    def test_points_that_are_not_a_projection_leave_a_large_residual(self) -> None:
        """This is the check ADR-0004 §5.4 asks for instead of looking at a
        rendered skeleton: a pinhole does not accidentally fit to a pixel."""
        three, two = self._correspondences()
        scrambled = two + np.random.default_rng(1).uniform(-60.0, 60.0, size=two.shape)

        assert recover_intrinsics(three, scrambled).residual_px > 10.0

    def test_points_behind_the_camera_are_refused(self) -> None:
        three = np.array([[0.0, 0.0, -1.0], [0.1, 0.1, 2.0]])
        two = np.array([[960.0, 540.0], [1000.0, 580.0]])

        with pytest.raises(GroundTruthError, match="behind the camera"):
            recover_intrinsics(three, two)

    def test_too_few_points_are_refused(self) -> None:
        with pytest.raises(GroundTruthError, match="at least two"):
            recover_intrinsics(np.zeros((1, 3)), np.zeros((1, 2)))
