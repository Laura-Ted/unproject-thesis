"""Tests for the pieces that turn a dataset and an artifact into observations.

The whole run needs a licensed dataset and a checkpoint, so no test runs it.
Everything reachable without them is tested here, because those are the parts
where a wrong answer would still look like a number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unproject.bodies.artifacts import CameraIntrinsics
from unproject.ground_truth.config import load_ground_truth_config
from unproject.ground_truth.dataset import (
    CANONICAL_CONVENTION,
    DatasetRecord,
    GroundTruthError,
    GroundTruthPerson,
)
from unproject.ground_truth.evaluate import (
    ROOT_JOINT,
    check_protocol,
    ground_truth_frame,
    project_to_image,
)
from unproject.ground_truth.mapping import load_mapping
from unproject.ground_truth.mupots import JOINT_NAMES

MAPPING_PATH = Path("experiments/ground-truth/mappings/mupots-3d-to-anny-v1.toml")
CONFIG_PATH = Path("experiments/ground-truth/mupots-3d.toml")


def _camera() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=1920, height=1080, fx=1100.0, fy=1100.0, cx=960.0, cy=540.0, source="estimated"
    )


class TestProjection:
    def test_a_point_on_the_axis_lands_on_the_optical_centre(self) -> None:
        joints = np.array([[0.0, 0.0, 3.0]])

        assert project_to_image(joints, _camera())[0] == pytest.approx([960.0, 540.0])

    def test_it_matches_the_pinhole_by_hand(self) -> None:
        joints = np.array([[0.3, -0.2, 3.0]])

        assert project_to_image(joints, _camera())[0] == pytest.approx(
            [1100.0 * 0.3 / 3.0 + 960.0, 1100.0 * -0.2 / 3.0 + 540.0]
        )

    def test_further_away_is_closer_to_the_centre(self) -> None:
        near = project_to_image(np.array([[0.5, 0.0, 2.0]]), _camera())
        far = project_to_image(np.array([[0.5, 0.0, 8.0]]), _camera())

        assert far[0][0] < near[0][0]

    def test_a_point_behind_the_camera_is_refused(self) -> None:
        """Projecting it would produce a finite pixel on the wrong side of the
        image, and matching would then compare a real person to a mirage."""
        with pytest.raises(GroundTruthError, match="behind the camera"):
            project_to_image(np.array([[0.1, 0.1, -3.0]]), _camera())

    def test_a_point_exactly_at_the_camera_is_refused(self) -> None:
        with pytest.raises(GroundTruthError, match="behind the camera"):
            project_to_image(np.array([[0.0, 0.0, 0.0]]), _camera())

    def test_it_projects_a_whole_frame_at_once(self) -> None:
        people = np.zeros((3, 14, 3))
        people[..., 2] = 4.0

        assert project_to_image(people, _camera()).shape == (3, 14, 2)


class TestReducingAFrame:
    def _record(self, people: int = 1, *, drop: int | None = None) -> DatasetRecord:
        joints = np.zeros((len(JOINT_NAMES), 3))
        joints[:, 2] = 3.0
        joints[:, 0] = np.linspace(-0.3, 0.3, len(JOINT_NAMES))
        two = np.stack(
            [np.linspace(800.0, 1100.0, len(JOINT_NAMES)), np.full(len(JOINT_NAMES), 500.0)], axis=1
        )
        valid = np.ones(len(JOINT_NAMES), dtype=np.bool_)
        if drop is not None:
            valid[drop] = False
        return DatasetRecord(
            dataset_name="mupots-3d",
            release_identity="fixture",
            sample_id="TS9/000000",
            sequence_id="TS9",
            frame_index=0,
            image_path=Path("TS9/img_000000.jpg"),
            image_width=1920,
            image_height=1080,
            people=tuple(
                GroundTruthPerson(
                    ground_truth_person_id=f"slot{index}",
                    joints_3d=joints + index * 0.5,
                    joint_names=JOINT_NAMES,
                    joint_valid=valid,
                    joints_2d=two + index * 200.0,
                    root_translation=joints[14],
                )
                for index in range(people)
            ),
            convention=CANONICAL_CONVENTION,
        )

    def test_seventeen_released_joints_become_the_fourteen_that_are_mapped(self) -> None:
        mapping, _ = load_mapping(MAPPING_PATH)

        three, two, valid = ground_truth_frame(self._record(), mapping)

        assert three.shape == (1, len(mapping.joints), 3)
        assert two.shape == (1, len(mapping.joints), 2)
        assert valid.all()

    def test_the_2d_reduction_matches_the_3d_reduction_joint_for_joint(self) -> None:
        """Both sides go through the same mapping, so a joint that moved in one
        and not the other would mean association and error disagreed about who
        a wrist is."""
        mapping, _ = load_mapping(MAPPING_PATH)
        record = self._record()
        pelvis = mapping.common_names.index(ROOT_JOINT)

        three, two, _ = ground_truth_frame(record, mapping)
        person = record.people[0]

        released = JOINT_NAMES.index("pelvis")
        assert person.joints_2d is not None
        assert three[0, pelvis] == pytest.approx(person.joints_3d[released])
        assert two[0, pelvis] == pytest.approx(person.joints_2d[released])

    def test_an_unannotated_released_joint_invalidates_only_what_reads_it(self) -> None:
        mapping, _ = load_mapping(MAPPING_PATH)
        dropped = JOINT_NAMES.index("left_wrist")

        _, _, valid = ground_truth_frame(self._record(drop=dropped), mapping)

        assert not valid[0, mapping.common_names.index("left_wrist")]
        assert valid[0, mapping.common_names.index("right_wrist")]

    def test_a_frame_with_nobody_in_it_reduces_to_nothing(self) -> None:
        mapping, _ = load_mapping(MAPPING_PATH)

        three, two, valid = ground_truth_frame(self._record(people=0), mapping)

        assert three.shape[0] == two.shape[0] == valid.shape[0] == 0

    def test_a_person_without_2d_is_refused(self) -> None:
        """Association would have to fall back on the 3D it is meant to test."""
        mapping, _ = load_mapping(MAPPING_PATH)
        record = self._record()
        without = DatasetRecord(
            **{
                **{field: getattr(record, field) for field in DatasetRecord.__slots__},
                "people": (
                    GroundTruthPerson(
                        ground_truth_person_id="slot0",
                        joints_3d=record.people[0].joints_3d,
                        joint_names=JOINT_NAMES,
                        joint_valid=record.people[0].joint_valid,
                        joints_2d=None,
                    ),
                ),
            }
        )

        with pytest.raises(GroundTruthError, match="no 2D joints"):
            ground_truth_frame(without, mapping)


class TestTheShippedConfiguration:
    def test_the_mapping_has_the_root_every_metric_is_defined_against(self) -> None:
        """`evaluate` refuses without it; this says the shipped one has it."""
        mapping, _ = load_mapping(MAPPING_PATH)

        assert ROOT_JOINT in mapping.common_names

    def test_the_matcher_is_part_of_the_configuration(self) -> None:
        """It decides who is compared with whom, so two runs that chose
        differently are two measurements and must not share an identity."""
        config = load_ground_truth_config(CONFIG_PATH)

        assert config.matcher == "mupots_greedy_2d_40px"
        assert "matcher" in config.model_dump_json()

    def test_the_shipped_configs_declare_the_protocol_and_the_camera_source(self) -> None:
        """ADR-0005 §3: a table names the protocol, and a camera says where it
        came from. Both conditions of the focal-length experiment ship."""
        control = load_ground_truth_config(CONFIG_PATH)
        treatment = load_ground_truth_config(
            Path("experiments/ground-truth/mupots-3d-annotation-derived-camera.toml")
        )

        assert control.metric_protocol == treatment.metric_protocol == "unproject_mupots_14j_v1"
        assert control.camera == "model_estimated"
        assert treatment.camera == "annotation_derived"
        assert control.thresholds.association_gate is None
        # Everything but the camera is the same, which is what makes the two a
        # control and a treatment.
        assert control.model_dump(
            exclude={"name", "description", "camera"}
        ) == treatment.model_dump(exclude={"name", "description", "camera"})

    def test_the_shipped_mapping_is_the_one_the_protocol_promises(self) -> None:
        mapping, _ = load_mapping(MAPPING_PATH)
        config = load_ground_truth_config(CONFIG_PATH)

        check_protocol(config, mapping)
        assert len(mapping.joints) == config.protocol.common_joints

    def test_a_mapping_of_the_wrong_size_cannot_hide_under_the_protocol_name(self) -> None:
        mapping, _ = load_mapping(MAPPING_PATH)
        config = load_ground_truth_config(CONFIG_PATH)
        smaller = mapping.model_copy(update={"joints": mapping.joints[:5]})

        with pytest.raises(GroundTruthError, match="different protocol"):
            check_protocol(config, smaller)
