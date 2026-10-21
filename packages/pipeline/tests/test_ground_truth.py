"""Tests for the ground-truth contracts, on data small enough to reason about.

No dataset, no weights, no network, no MPS. Everything here is hand-authored so
that when a check fails it is obvious what it was protecting.

The recurring theme: every one of these mistakes produces a *plausible* number
rather than a crash. Millimetres read as metres, a mapping matched by position,
a person quietly dropped from an average — none of them look wrong in a table.
That is why they fail closed here instead.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from fixture_types import WriteConfig
from unproject.ground_truth.artifacts import (
    EVALUATION_SCHEMA_VERSION,
    Coverage,
    DatasetIdentity,
    Environment,
    EvaluationDescriptor,
    MatchingIdentity,
    describe_file,
    matching_identity,
    read_descriptor,
    staged_evaluation,
    write_descriptor,
)
from unproject.ground_truth.config import (
    GroundTruthConfig,
    MetricThresholds,
    identity_hash,
    load_ground_truth_config,
)
from unproject.ground_truth.dataset import (
    CANONICAL_CONVENTION,
    Camera,
    DatasetRecord,
    GroundTruthError,
    GroundTruthPerson,
    validate_record,
)
from unproject.ground_truth.inventory import inventory, read_subset_manifest
from unproject.ground_truth.mapping import (
    JointMapping,
    apply_mapping,
    load_mapping,
)

_CHECKPOINT = "f" * 64


def _person() -> GroundTruthPerson:
    """One person standing five metres away, joints in metres."""
    return GroundTruthPerson(
        ground_truth_person_id="p0",
        joints_3d=np.array([[-0.1, 0.0, 5.0], [0.1, 0.0, 5.0], [-0.1, 0.5, 5.0]]),
        joint_names=("hip_l", "hip_r", "knee_l"),
        joint_valid=np.ones(3, dtype=np.bool_),
    )


def _record() -> DatasetRecord:
    """One frame of one MuPoTS-shaped sequence. Vary it with `replace`."""
    return DatasetRecord(
        dataset_name="mupots-3d",
        release_identity="test",
        sample_id="TS1/0001",
        sequence_id="TS1",
        frame_index=1,
        image_path=Path("TS1/img_000001.jpg"),
        image_width=2048,
        image_height=2048,
        people=(_person(),),
        camera=Camera(fx=1500.0, fy=1500.0, cx=1024.0, cy=1024.0),
    )


class TestTheNormalisedContract:
    def test_a_well_formed_record_passes(self) -> None:
        assert validate_record(_record(), source="fixture") is not None

    def test_a_foreign_convention_is_refused(self) -> None:
        """A root-relative annotation read as camera-space is the whole risk."""
        with pytest.raises(GroundTruthError, match=re.escape(CANONICAL_CONVENTION)):
            validate_record(replace(_record(), convention="world_metres"), source="fixture")

    def test_millimetres_left_unconverted_are_caught(self) -> None:
        # 5000 mm read as 5000 m. Nothing about the array says which it is, so
        # the only defence is a bound on what a person's distance can be.
        millimetres = replace(_person(), joints_3d=np.array([[-100.0, 0.0, 5000.0]] * 3))
        with pytest.raises(GroundTruthError, match="millimetre"):
            validate_record(replace(_record(), people=(millimetres,)), source="fixture")

    def test_a_validity_mask_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(GroundTruthError, match="same length"):
            validate_record(
                replace(
                    _record(), people=(replace(_person(), joint_valid=np.ones(2, dtype=np.bool_)),)
                ),
                source="fixture",
            )

    def test_a_repeated_joint_name_is_refused(self) -> None:
        """Names address joints in the mapping; a duplicate silently merges two."""
        with pytest.raises(GroundTruthError, match="repeats a joint name"):
            validate_record(
                replace(
                    _record(),
                    people=(replace(_person(), joint_names=("hip_l", "hip_l", "knee_l")),),
                ),
                source="fixture",
            )

    def test_a_non_finite_valid_joint_is_refused(self) -> None:
        broken = replace(
            _person(), joints_3d=np.array([[np.nan, 0.0, 5.0], [0.1, 0.0, 5.0], [0.0, 0.5, 5.0]])
        )
        with pytest.raises(GroundTruthError, match="non-finite"):
            validate_record(replace(_record(), people=(broken,)), source="fixture")

    def test_a_non_finite_joint_marked_invalid_is_fine(self) -> None:
        """A release may park an unannotated joint at NaN; dropping it is correct."""
        parked = replace(
            _person(),
            joints_3d=np.array([[np.nan, np.nan, np.nan], [0.1, 0.0, 5.0], [0.0, 0.5, 5.0]]),
            joint_valid=np.array([False, True, True]),
        )

        assert validate_record(replace(_record(), people=(parked,)), source="fixture") is not None

    def test_an_integer_joint_array_is_refused(self) -> None:
        """An integer array here means a unit was truncated upstream."""
        with pytest.raises(GroundTruthError, match="float array"):
            validate_record(
                replace(
                    _record(),
                    people=(replace(_person(), joints_3d=np.zeros((3, 3), dtype=np.int64)),),
                ),
                source="fixture",
            )

    def test_an_error_names_the_source_it_came_from(self) -> None:
        with pytest.raises(GroundTruthError, match=r"TS1\.mat"):
            validate_record(replace(_record(), convention="wrong"), source="TS1.mat")

    def test_a_frame_with_nobody_in_it_is_valid(self) -> None:
        """An empty frame is a real observation; it becomes recall, not an error."""
        assert validate_record(replace(_record(), people=()), source="fixture") is not None


def _mapping_toml(**overrides: str) -> str:
    fields = {
        "identifier": "test-v1",
        "anny_package_version": "0.6.0",
        "anny_bone_count": "163",
        "checkpoint_sha256": f'"{_CHECKPOINT}"',
    }
    fields.update(overrides)
    return (
        "schema_version = 1\n"
        f'identifier = "{fields["identifier"]}"\n'
        f'anny_package_version = "{fields["anny_package_version"]}"\n'
        f"anny_bone_count = {fields['anny_bone_count']}\n"
        f"checkpoint_sha256 = {fields['checkpoint_sha256']}\n"
        "[[joints]]\n"
        'name = "pelvis"\n'
        'provenance = "midpoint of the two hips; neither release defines a pelvis"\n'
        'mupots = { names = ["hip_l", "hip_r"], derivation = "midpoint" }\n'
        'anny = { names = ["LHipJoint", "RHipJoint"], derivation = "midpoint" }\n'
        "[[joints]]\n"
        'name = "left_knee"\n'
        'provenance = "same anatomical joint, both released as a knee"\n'
        'mupots = { names = ["knee_l"] }\n'
        'anny = { names = ["LeftLeg"] }\n'
    )


class TestTheJointMapping:
    def test_it_reads_a_whole_mapping(self, write_config: WriteConfig) -> None:
        path = write_config(_mapping_toml(), name="m.toml")

        mapping, digest = load_mapping(path)

        assert mapping.common_names == ("pelvis", "left_knee")
        assert len(digest) == 64

    def test_the_digest_changes_with_the_file(self, write_config: WriteConfig) -> None:
        """Two results under different correspondences must not look comparable."""
        first = load_mapping(write_config(_mapping_toml(), name="a.toml"))[1]
        second = load_mapping(write_config(_mapping_toml(identifier="test-v2"), name="b.toml"))[1]

        assert first != second

    def test_a_duplicate_common_joint_is_refused(self, write_config: WriteConfig) -> None:
        doubled = _mapping_toml() + (
            "[[joints]]\n"
            'name = "pelvis"\n'
            'provenance = "again"\n'
            'mupots = { names = ["hip_l"] }\n'
            'anny = { names = ["LHipJoint"] }\n'
        )
        with pytest.raises(GroundTruthError, match="more than once"):
            load_mapping(write_config(doubled, name="m.toml"))

    def test_a_direct_joint_with_two_sources_is_refused(self, write_config: WriteConfig) -> None:
        broken = _mapping_toml().replace(
            'mupots = { names = ["knee_l"] }', 'mupots = { names = ["knee_l", "knee_r"] }'
        )
        with pytest.raises(GroundTruthError, match="exactly one source name"):
            load_mapping(write_config(broken, name="m.toml"))

    def test_a_midpoint_with_one_source_is_refused(self, write_config: WriteConfig) -> None:
        broken = _mapping_toml().replace(
            'mupots = { names = ["hip_l", "hip_r"], derivation = "midpoint" }',
            'mupots = { names = ["hip_l"], derivation = "midpoint" }',
        )
        with pytest.raises(GroundTruthError, match="at least two source names"):
            load_mapping(write_config(broken, name="m.toml"))

    def test_an_unreadable_schema_version_is_refused(self, write_config: WriteConfig) -> None:
        with pytest.raises(GroundTruthError, match="schema_version"):
            load_mapping(
                write_config(_mapping_toml().replace("schema_version = 1", "schema_version = 9"))
            )

    def test_it_maps_direct_and_derived_joints(self, write_config: WriteConfig) -> None:
        mapping, _ = load_mapping(write_config(_mapping_toml(), name="m.toml"))
        names = ("hip_l", "hip_r", "knee_l")
        joints = np.array([[-0.1, 0.0, 5.0], [0.1, 0.0, 5.0], [-0.1, 0.5, 5.0]])

        common, valid = apply_mapping(mapping, "mupots", joints, names, np.ones(3, dtype=np.bool_))

        assert bool(valid.all())
        # The pelvis is the midpoint of the two hips, so it lands between them.
        assert common[0] == pytest.approx([0.0, 0.0, 5.0])
        assert common[1] == pytest.approx([-0.1, 0.5, 5.0])

    def test_a_derived_joint_needs_every_source_to_be_valid(
        self, write_config: WriteConfig
    ) -> None:
        """Half a pelvis is not a pelvis, and averaging it in would be an invention."""
        mapping, _ = load_mapping(write_config(_mapping_toml(), name="m.toml"))
        names = ("hip_l", "hip_r", "knee_l")
        joints = np.array([[-0.1, 0.0, 5.0], [0.1, 0.0, 5.0], [-0.1, 0.5, 5.0]])

        _, valid = apply_mapping(mapping, "mupots", joints, names, np.array([True, False, True]))

        assert not bool(valid[0])
        assert bool(valid[1])

    def test_a_release_missing_a_mapped_joint_fails_closed(self, write_config: WriteConfig) -> None:
        """The message has to name what it wanted; matching by position is the sin."""
        mapping, _ = load_mapping(write_config(_mapping_toml(), name="m.toml"))

        with pytest.raises(GroundTruthError, match="do not match by position"):
            apply_mapping(
                mapping,
                "mupots",
                np.zeros((2, 3)),
                ("hip_l", "hip_r"),
                np.ones(2, dtype=np.bool_),
            )


_CONFIG = """
schema_version = 2
name = "mupots-test"
dataset = "mupots-3d"
dataset_root = "../datasets/mupots-3d"
release = "downloaded 2026-09-08"
experiment = "../crowd.toml"
mapping = "mappings/m.toml"
"""

_CONFIG_V1 = """
schema_version = 1
name = "mupots-test"
dataset = "mupots-3d"
dataset_root = "../datasets/mupots-3d"
release = "downloaded 2026-09-08"
experiment = "../crowd.toml"
mapping = "mappings/m.toml"
matcher = "official"
camera = "released"
[thresholds]
association_gate = 0.1
"""


class TestTheConfig:
    def test_it_reads_a_whole_config(self, write_config: WriteConfig) -> None:
        config = load_ground_truth_config(write_config(_CONFIG, name="gt.toml"))

        assert config.dataset == "mupots-3d"
        assert config.thresholds.pck_radius_mm == 150.0
        # The defaults say what they are. ADR-0005 §3.
        assert config.metric_protocol == "unproject_mupots_14j_v1"
        assert config.matcher == "mupots_greedy_2d_40px"
        assert config.camera == "model_estimated"

    def test_a_schema_1_file_is_refused_with_the_rename_table(
        self, write_config: WriteConfig
    ) -> None:
        """Not read under the old names: `official` and `released` are the
        words the file has to stop using, and reading them silently would keep
        them alive in every artifact the file produces."""
        with pytest.raises(GroundTruthError) as caught:
            load_ground_truth_config(write_config(_CONFIG_V1, name="gt.toml"))

        message = str(caught.value)
        assert "schema 1" in message
        assert 'matcher = "official"  ->  matcher = "mupots_greedy_2d_40px"' in message
        assert 'camera = "released"  ->  camera = "annotation_derived"' in message
        assert "metric_protocol" in message

    def test_the_old_names_are_not_accepted_under_schema_2(self, write_config: WriteConfig) -> None:
        with pytest.raises(GroundTruthError, match="matcher"):
            load_ground_truth_config(
                write_config(_CONFIG + 'matcher = "official"\n', name="gt.toml")
            )
        with pytest.raises(GroundTruthError, match="camera"):
            load_ground_truth_config(
                write_config(_CONFIG + 'camera = "released"\n', name="gt.toml")
            )

    def test_the_greedy_matcher_refuses_a_gate_it_would_never_read(
        self, write_config: WriteConfig
    ) -> None:
        """v1 recorded `association_gate = 0.1` beside greedy results the number
        never touched. A setting that does nothing must not be recordable."""
        body = _CONFIG + "\n[thresholds]\nassociation_gate = 0.1\n"
        with pytest.raises(GroundTruthError, match="does nothing"):
            load_ground_truth_config(write_config(body, name="gt.toml"))

    def test_the_global_matcher_requires_its_gate(self, write_config: WriteConfig) -> None:
        body = _CONFIG + 'matcher = "normalized_global_assignment"\n'
        with pytest.raises(GroundTruthError, match=r"needs thresholds\.association_gate"):
            load_ground_truth_config(write_config(body, name="gt.toml"))

    def test_the_global_matcher_reads_its_gate(self, write_config: WriteConfig) -> None:
        body = (
            _CONFIG + 'matcher = "normalized_global_assignment"\n'
            "[thresholds]\nassociation_gate = 0.05\n"
        )
        config = load_ground_truth_config(write_config(body, name="gt.toml"))

        assert config.thresholds.association_gate == 0.05

    def test_the_camera_source_is_part_of_the_identity(self) -> None:
        """Control and treatment must never share a directory."""
        base = GroundTruthConfig.model_validate(
            {
                "name": "x",
                "dataset": "mupots-3d",
                "dataset_root": "/place",
                "release": "r",
                "experiment": "e.toml",
                "mapping": "m.toml",
            }
        )
        told = base.model_copy(update={"camera": "annotation_derived"})

        assert identity_hash(base, inventory_hash="a" * 64, mapping_hash="b" * 64) != identity_hash(
            told, inventory_hash="a" * 64, mapping_hash="b" * 64
        )

    def test_a_typo_is_refused_rather_than_defaulted(self, write_config: WriteConfig) -> None:
        with pytest.raises(GroundTruthError, match="pck_radius"):
            load_ground_truth_config(
                write_config(_CONFIG + "\n[thresholds]\npck_radius = 150.0\n", name="gt.toml")
            )

    def test_anny_one_without_a_subset_is_refused(self, write_config: WriteConfig) -> None:
        """780,964 images is not a default anyone should reach by omission."""
        body = _CONFIG.replace('dataset = "mupots-3d"', 'dataset = "anny-one"')
        with pytest.raises(GroundTruthError, match="subset_manifest"):
            load_ground_truth_config(write_config(body, name="gt.toml"))

    def test_depth_bins_must_ascend(self, write_config: WriteConfig) -> None:
        body = _CONFIG + "\n[thresholds]\ndepth_bins_metres = [4.0, 2.0]\n"
        with pytest.raises(GroundTruthError, match="ascend"):
            load_ground_truth_config(write_config(body, name="gt.toml"))

    def test_moving_the_dataset_does_not_change_identity(self) -> None:
        """A path says where something is; content says what it is. ADR-0002 §7.2."""
        here = GroundTruthConfig.model_validate(
            {
                "name": "x",
                "dataset": "mupots-3d",
                "dataset_root": "/one/place",
                "release": "r",
                "experiment": "e.toml",
                "mapping": "m.toml",
            }
        )
        there = here.model_copy(update={"dataset_root": Path("/somewhere/else")})

        assert identity_hash(here, inventory_hash="a" * 64, mapping_hash="b" * 64) == identity_hash(
            there, inventory_hash="a" * 64, mapping_hash="b" * 64
        )

    def test_changing_the_dataset_contents_does_change_identity(self) -> None:
        config = GroundTruthConfig.model_validate(
            {
                "name": "x",
                "dataset": "mupots-3d",
                "dataset_root": "/one/place",
                "release": "r",
                "experiment": "e.toml",
                "mapping": "m.toml",
            }
        )

        assert identity_hash(
            config, inventory_hash="a" * 64, mapping_hash="b" * 64
        ) != identity_hash(config, inventory_hash="c" * 64, mapping_hash="b" * 64)

    def test_changing_a_threshold_changes_identity(self) -> None:
        base = GroundTruthConfig.model_validate(
            {
                "name": "x",
                "dataset": "mupots-3d",
                "dataset_root": "/one/place",
                "release": "r",
                "experiment": "e.toml",
                "mapping": "m.toml",
            }
        )
        moved = base.model_copy(update={"thresholds": MetricThresholds(pck_radius_mm=100.0)})

        assert identity_hash(base, inventory_hash="a" * 64, mapping_hash="b" * 64) != identity_hash(
            moved, inventory_hash="a" * 64, mapping_hash="b" * 64
        )


class TestTheInventory:
    def _config(self, root: Path) -> GroundTruthConfig:
        return GroundTruthConfig.model_validate(
            {
                "name": "x",
                "dataset": "mupots-3d",
                "dataset_root": str(root),
                "release": "r",
                "experiment": "e.toml",
                "mapping": "m.toml",
            }
        )

    def test_a_missing_dataset_says_where_to_get_it(self, tmp_path: Path) -> None:
        """The licence is the author's to accept, so the error is for a person."""
        with pytest.raises(GroundTruthError, match=r"vcai\.mpi-inf\.mpg\.de"):
            inventory(self._config(tmp_path / "nope"), dataset_root=tmp_path / "nope")

    def test_a_dataset_with_no_annotations_is_refused(self, tmp_path: Path) -> None:
        """Otherwise the evaluation reports perfect coverage of zero people."""
        root = tmp_path / "mupots"
        (root / "images").mkdir(parents=True)
        (root / "images" / "a.jpg").write_bytes(b"not an annotation")

        with pytest.raises(GroundTruthError, match="no annotation files"):
            inventory(self._config(root), dataset_root=root)

    def test_it_identifies_the_release_by_content(self, tmp_path: Path) -> None:
        root = tmp_path / "mupots"
        root.mkdir()
        (root / "annot.mat").write_bytes(b"one")

        before = inventory(self._config(root), dataset_root=root)
        (root / "annot.mat").write_bytes(b"two")
        after = inventory(self._config(root), dataset_root=root)

        assert before.content_sha256 != after.content_sha256
        assert before.file_count == 1

    def test_moving_a_file_within_the_release_changes_identity(self, tmp_path: Path) -> None:
        root = tmp_path / "mupots"
        (root / "a").mkdir(parents=True)
        (root / "b").mkdir()
        (root / "a" / "annot.mat").write_bytes(b"same bytes")
        before = inventory(self._config(root), dataset_root=root)

        (root / "a" / "annot.mat").rename(root / "b" / "annot.mat")
        after = inventory(self._config(root), dataset_root=root)

        assert before.content_sha256 != after.content_sha256


class TestTheSubsetManifest:
    def test_it_reads_identifiers_and_ignores_comments(self, tmp_path: Path) -> None:
        path = tmp_path / "subset.txt"
        path.write_text("# frozen 2026-09-08\nscene_0001\nscene_0002\n\n", encoding="utf-8")

        identifiers, digest = read_subset_manifest(path)

        assert identifiers == ("scene_0001", "scene_0002")
        assert len(digest) == 64

    def test_a_repeated_identifier_is_refused(self, tmp_path: Path) -> None:
        """A sample listed twice is weighted twice in every average."""
        path = tmp_path / "subset.txt"
        path.write_text("scene_0001\nscene_0001\n", encoding="utf-8")

        with pytest.raises(GroundTruthError, match="more than once"):
            read_subset_manifest(path)

    def test_an_unsorted_manifest_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "subset.txt"
        path.write_text("scene_0002\nscene_0001\n", encoding="utf-8")

        with pytest.raises(GroundTruthError, match="not sorted"):
            read_subset_manifest(path)

    def test_an_empty_manifest_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "subset.txt"
        path.write_text("# nothing yet\n", encoding="utf-8")

        with pytest.raises(GroundTruthError, match="no sample identifiers"):
            read_subset_manifest(path)


def _descriptor(**overrides: object) -> EvaluationDescriptor:
    started = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
    defaults: dict[str, object] = {
        "evaluation_id": "mupots-test-abc123",
        "started_at": started,
        "finished_at": started,
        "config": GroundTruthConfig.model_validate(
            {
                "name": "x",
                "dataset": "mupots-3d",
                "dataset_root": "/place",
                "release": "r",
                "experiment": "e.toml",
                "mapping": "m.toml",
            }
        ),
        "config_hash": "a" * 64,
        "mapping_identifier": "test-v1",
        "mapping_sha256": "b" * 64,
        "dataset": DatasetIdentity(
            name="mupots-3d",
            release="r",
            split="test",
            inventory_sha256="c" * 64,
            file_count=1,
            byte_count=10,
        ),
        "environment": Environment(
            machine="Apple M3 Pro",
            platform="Darwin 25.6.0 arm64",
            python_version="3.12.14",
            device="mps",
            package_versions={},
        ),
        "metric_protocol": "unproject_mupots_14j_v1",
        "matching": MatchingIdentity(
            matcher="mupots_greedy_2d_40px", parameter="radius_px_per_axis", value=40.0
        ),
        "camera_source": "model_estimated",
        "checkpoint_sha256": _CHECKPOINT,
        "coverage": Coverage(
            samples=1,
            ground_truth_people=2,
            predictions=2,
            matched=2,
            false_negatives=0,
            false_positives=0,
            valid_joints=6,
        ),
    }
    return EvaluationDescriptor.model_validate(defaults | overrides)


class TestCoverage:
    def test_the_books_have_to_balance(self) -> None:
        """A person dropped between annotation and matching is exactly the bug."""
        with pytest.raises(ValueError, match="a person was dropped"):
            Coverage(
                samples=1,
                ground_truth_people=10,
                predictions=1,
                matched=1,
                false_negatives=0,
                false_positives=0,
                valid_joints=3,
            )

    def test_missing_people_are_representable(self) -> None:
        """Finding one of ten is a valid result. It must not read as accurate."""
        coverage = Coverage(
            samples=1,
            ground_truth_people=10,
            predictions=1,
            matched=1,
            false_negatives=9,
            false_positives=0,
            valid_joints=3,
        )

        assert coverage.false_negatives == 9


class TestPublication:
    def test_a_finished_evaluation_reads_back(self, tmp_path: Path) -> None:
        target = tmp_path / "eval"
        with staged_evaluation(target) as staging:
            (staging / "observations.npz").write_bytes(b"numbers")
            reference = describe_file(staging / "observations.npz", relative_to=staging)
            write_descriptor(_descriptor(references=[reference]), staging)

        assert read_descriptor(target).evaluation_id == "mupots-test-abc123"

    def test_nothing_lands_when_the_run_fails(self, tmp_path: Path) -> None:
        target = tmp_path / "eval"

        with pytest.raises(RuntimeError), staged_evaluation(target) as staging:
            (staging / "observations.npz").write_bytes(b"partial")
            raise RuntimeError("inference died")

        # The descriptor is the completion marker; without it nothing claims to
        # be a finished evaluation.
        assert not (target / "evaluation.json").exists()

    def test_a_truncated_sidecar_is_caught(self, tmp_path: Path) -> None:
        """A descriptor that vouches for a file it cannot is worse than none."""
        target = tmp_path / "eval"
        with staged_evaluation(target) as staging:
            (staging / "observations.npz").write_bytes(b"numbers")
            reference = describe_file(staging / "observations.npz", relative_to=staging)
            write_descriptor(_descriptor(references=[reference]), staging)

        (target / "observations.npz").write_bytes(b"cut")

        with pytest.raises(GroundTruthError, match="truncated or replaced"):
            read_descriptor(target)

    def test_a_missing_sidecar_is_caught(self, tmp_path: Path) -> None:
        target = tmp_path / "eval"
        with staged_evaluation(target) as staging:
            (staging / "observations.npz").write_bytes(b"numbers")
            reference = describe_file(staging / "observations.npz", relative_to=staging)
            write_descriptor(_descriptor(references=[reference]), staging)

        (target / "observations.npz").unlink()

        with pytest.raises(GroundTruthError, match="is not there"):
            read_descriptor(target)

    def test_an_unreadable_schema_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            _descriptor(schema_version=EVALUATION_SCHEMA_VERSION + 1)

    def test_the_labels_repeated_for_readers_cannot_disagree_with_the_config(self) -> None:
        with pytest.raises(ValueError, match="disagree"):
            _descriptor(camera_source="annotation_derived")

    def test_the_matching_identity_names_the_parameter_that_was_read(self) -> None:
        greedy = GroundTruthConfig.model_validate(
            {
                "name": "x",
                "dataset": "mupots-3d",
                "dataset_root": "/place",
                "release": "r",
                "experiment": "e.toml",
                "mapping": "m.toml",
            }
        )
        global_ = greedy.model_copy(
            update={
                "matcher": "normalized_global_assignment",
                "thresholds": MetricThresholds(association_gate=0.07),
            }
        )

        assert matching_identity(greedy) == MatchingIdentity(
            matcher="mupots_greedy_2d_40px", parameter="radius_px_per_axis", value=40.0
        )
        assert matching_identity(global_) == MatchingIdentity(
            matcher="normalized_global_assignment",
            parameter="gate_fraction_of_image_diagonal",
            value=0.07,
        )


_V1_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "name": "mupots-3d-test-released-camera",
    "description": "",
    "dataset": "mupots-3d",
    "dataset_root": "../../datasets/mupots-3d",
    "release": "r",
    "split": "test",
    "subset_manifest": None,
    "experiment": "mupots-estimator.toml",
    "mapping": "mappings/mupots-3d-to-anny-v1.toml",
    "matcher": "official",
    "camera": "released",
    "thresholds": {
        "pck_radius_mm": 150.0,
        "association_gate": 0.1,
        "depth_bins_metres": [2.5, 3.5, 4.5, 6.0],
        "max_temporal_interval_seconds": 0.2,
    },
    "output_root": "../../artifacts/ground-truth",
    "seed": 0,
    "official_evaluator": None,
}

_V1_DESCRIPTOR: dict[str, object] = {
    "schema_version": 1,
    "evaluation_id": "a1a82f0677ec0000",
    "started_at": "2026-09-11T10:00:00Z",
    "finished_at": "2026-09-11T10:01:00Z",
    "status": "ok",
    "error": None,
    "config": _V1_CONFIG,
    "config_hash": "a" * 64,
    "mapping_identifier": "mupots-3d-to-anny-v1",
    "mapping_sha256": "b" * 64,
    "dataset": {
        "name": "mupots-3d",
        "release": "r",
        "split": "test",
        "inventory_sha256": "c" * 64,
        "file_count": 20,
        "byte_count": 10,
        "subset_manifest_sha256": None,
        "excluded": [],
        "exclusion_reason": "",
    },
    "environment": {
        "machine": "arm64",
        "platform": "Darwin 25.6.0 arm64",
        "python_version": "3.12.14",
        "device": "mps",
        "package_versions": {},
        "git_commit": "27e0ec5",
        "git_dirty": True,
    },
    "body_artifact_hashes": ["0123456789ab", "ba9876543210"],
    "checkpoint_sha256": _CHECKPOINT,
    "coverage": {
        "samples": 1,
        "ground_truth_people": 2,
        "predictions": 2,
        "matched": 2,
        "false_negatives": 0,
        "false_positives": 0,
        "valid_joints": 6,
    },
    "references": [],
    "failures": [],
}


class TestReadingAVersionOneDescriptor:
    """The two thesis evaluations were written under schema 1. They stay
    readable, under the names they should have had, and say they were
    upgraded. ADR-0005 WP1: labels change, numbers do not."""

    def _write(self, tmp_path: Path) -> Path:
        import json

        directory = tmp_path / "old"
        directory.mkdir()
        (directory / "evaluation.json").write_text(json.dumps(_V1_DESCRIPTOR), encoding="utf-8")
        return directory

    def test_the_labels_are_upgraded(self, tmp_path: Path) -> None:
        descriptor = read_descriptor(self._write(tmp_path))

        assert descriptor.schema_version == EVALUATION_SCHEMA_VERSION
        assert descriptor.config.matcher == "mupots_greedy_2d_40px"
        assert descriptor.config.camera == "annotation_derived"
        assert descriptor.camera_source == "annotation_derived"
        assert descriptor.metric_protocol == "unproject_mupots_14j_v1"
        assert descriptor.matching.parameter == "radius_px_per_axis"

    def test_the_gate_the_greedy_matcher_never_read_is_dropped_and_said_so(
        self, tmp_path: Path
    ) -> None:
        descriptor = read_descriptor(self._write(tmp_path))

        assert descriptor.config.thresholds.association_gate is None
        assert descriptor.migration is not None
        assert descriptor.migration.from_schema_version == 1
        assert any("association_gate 0.1 dropped" in note for note in descriptor.migration.notes)

    def test_body_artifacts_keep_their_hashes_with_unknown_git_state(self, tmp_path: Path) -> None:
        """v1 never read the manifests, so the state is unknown, not clean."""
        descriptor = read_descriptor(self._write(tmp_path))

        assert descriptor.body_artifact_hashes == ["0123456789ab", "ba9876543210"]
        assert all(artifact.git_dirty is None for artifact in descriptor.body_artifacts)
        assert descriptor.final is False

    def test_a_native_v2_file_is_not_marked_as_migrated(self, tmp_path: Path) -> None:
        target = tmp_path / "new"
        with staged_evaluation(target) as staging:
            write_descriptor(_descriptor(), staging)

        assert read_descriptor(target).migration is None

    def test_an_unknown_v1_matcher_is_refused(self, tmp_path: Path) -> None:
        import json

        broken = {**_V1_DESCRIPTOR, "config": {**_V1_CONFIG, "matcher": "nearest"}}
        directory = tmp_path / "broken"
        directory.mkdir()
        (directory / "evaluation.json").write_text(json.dumps(broken), encoding="utf-8")

        with pytest.raises(GroundTruthError, match="no matcher 'nearest'"):
            read_descriptor(directory)


def test_the_package_loads_without_torch() -> None:
    """ADR-0004 WP1: validation runs without model weights.

    A reader checking a dataset should not pay three gigabytes of inference
    stack to be told a file is missing. Checked in a subprocess because this
    one has almost certainly imported torch already.
    """
    source = (
        "import sys; import unproject.ground_truth as gt; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m); "
        "print(gt.CANONICAL_CONVENTION)"
    )
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "camera_relative_opencv_metres"


class TestTheCommittedMapping:
    """The real mapping, checked as data rather than trusted as prose.

    Both sides were sourced rather than inferred, and getting there passed two
    plausible wrong answers - see docs/notes/2026-09-10-joint-mapping-sourced.md.
    These pin the parts a future edit could quietly break.
    """

    def _mapping(self) -> tuple[JointMapping, str]:
        root = Path(__file__).resolve().parents[3]
        return load_mapping(root / "experiments/ground-truth/mappings/mupots-3d-to-anny-v1.toml")

    def test_it_loads_and_declares_the_model_it_was_written_against(self) -> None:
        mapping, _ = self._mapping()

        # 163, not the 104 of anny's own default rig. The two orderings differ
        # and mapping through the wrong one shifts every arm joint by two.
        assert mapping.anny_bone_count == 163
        assert mapping.anny_package_version == "0.6.0"

    def test_the_pelvis_is_anny_bone_zero(self) -> None:
        """Root alignment and every absolute metric are defined against it."""
        mapping, _ = self._mapping()
        [pelvis] = [joint for joint in mapping.joints if joint.name == "pelvis"]

        assert pelvis.anny.names == ["root"]
        assert pelvis.mupots.names == ["pelvis"]

    def test_limb_joints_name_the_bone_that_starts_at_them(self) -> None:
        """j3d is a bone's origin, so the knee is where the shin begins."""
        mapping, _ = self._mapping()
        anny = {joint.name: joint.anny.names[0] for joint in mapping.joints}

        assert anny["left_knee"] == "lowerleg01.L"
        assert anny["left_ankle"] == "foot.L"
        assert anny["left_elbow"] == "lowerarm01.L"
        assert anny["left_hip"] == "upperleg01.L"

    def test_the_ambiguous_joints_are_absent(self) -> None:
        """head_top, head and spine have no unambiguous Anny counterpart."""
        mapping, _ = self._mapping()

        assert set(mapping.common_names).isdisjoint({"head_top", "head", "spine"})

    def test_it_is_left_and_right_symmetric(self) -> None:
        """A mapping that gained a side would be a mapping someone half-edited."""
        mapping, _ = self._mapping()
        names = set(mapping.common_names)
        left = {name for name in names if name.startswith("left_")}

        assert {name.replace("left_", "right_") for name in left} <= names

    def test_it_reduces_a_full_anny_skeleton(self) -> None:
        mapping, _ = self._mapping()
        labels = [f"bone{index}" for index in range(163)]
        for index, name in (
            (0, "root"),
            (2, "upperleg01.L"),
            (4, "lowerleg01.L"),
            (6, "foot.L"),
            (22, "upperleg01.R"),
            (24, "lowerleg01.R"),
            (26, "foot.R"),
            (50, "upperarm01.L"),
            (52, "lowerarm01.L"),
            (54, "wrist.L"),
            (76, "upperarm01.R"),
            (78, "lowerarm01.R"),
            (80, "wrist.R"),
            (100, "neck01"),
        ):
            labels[index] = name

        common, valid = apply_mapping(
            mapping,
            "anny",
            np.arange(163 * 3, dtype=np.float64).reshape(163, 3) / 1000.0,
            tuple(labels),
            np.ones(163, dtype=np.bool_),
        )

        assert common.shape == (14, 3)
        assert bool(valid.all())

    def test_a_skeleton_missing_one_mapped_bone_fails_closed(self) -> None:
        mapping, _ = self._mapping()
        labels = [f"bone{index}" for index in range(163)]

        with pytest.raises(GroundTruthError, match="do not match by position"):
            apply_mapping(
                mapping,
                "anny",
                np.zeros((163, 3)),
                tuple(labels),
                np.ones(163, dtype=np.bool_),
            )
