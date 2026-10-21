"""Tests for running a dataset sequence through stage [B].

Stage [B] itself is not exercised here - it needs a checkpoint and a GPU. What
is exercised is the two things that decide whether a later metric means
anything: that each sequence gets its own experiment, and that a sequence whose
frame count disagrees with its annotations is refused rather than scored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unproject.bodies.artifacts import (
    BodiesArtifact,
    BodyModelInfo,
    DetectionSummary,
    EstimatorInfo,
    ParameterArray,
    ParameterArrays,
)
from unproject.bodies.stage import BodiesResult
from unproject.config import ExperimentConfig
from unproject.ground_truth import sequences as sequences_module
from unproject.ground_truth.dataset import Camera, DatasetRecord, GroundTruthError
from unproject.ground_truth.mupots import JOINT_NAMES, SequenceIntrinsics
from unproject.ground_truth.sequences import (
    SequenceRun,
    annotation_derived_camera,
    release_frames,
    run_sequence,
    sequence_experiment,
)

_DIGEST = "fd2d9ab7010a5f590a3db2480f5f476fd3ec6afbbde80babbec0d821ba9763d6"


def _experiment(**ingest: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "mupots",
            "ingest": {"source_path": "frames", "image_fps": 30.0, **ingest},
        }
    )


def _artifact(frames_total: int) -> BodiesArtifact:
    """An artifact that claims a frame count and nothing else worth asserting."""
    return BodiesArtifact.model_validate(
        {
            "ingest_hash": "0140aa2c6860",
            "body_model": BodyModelInfo(
                name="anny", version="0.6.0", bone_count=2, shape_count=11, bone_parents=[-1, 0]
            ),
            "estimator": EstimatorInfo(
                name="multihmr2",
                checkpoint_sha256=_DIGEST,
                backbone_repo="facebookresearch/dinov3",
                backbone_ref="6876159a11b4df116f30f667f8c9888617df0751",
                config_hash="deadbeef",
            ),
            "parameters": ParameterArrays(
                path="bodies.bin",
                person_count=0,
                arrays=[
                    ParameterArray(
                        name="root_translation", shape=(0, 3), byte_offset=0, byte_length=0
                    )
                ],
            ),
            "frames": [
                {
                    "frame_index": index,
                    "source_index": index,
                    "timestamp_seconds": index / 30.0,
                    "camera": {
                        "width": 1920,
                        "height": 1080,
                        "fx": 1100.0,
                        "fy": 1100.0,
                        "cx": 960.0,
                        "cy": 540.0,
                        "source": "estimated",
                    },
                    "people": [],
                }
                for index in range(frames_total)
            ],
            "summary": DetectionSummary(
                frames_total=frames_total,
                frames_with_people=0,
                people_total=0,
                max_people_in_frame=0,
                confidence_threshold=0.4,
                track_count=0,
            ),
        }
    )


def _records(count: int) -> list[DatasetRecord]:
    return [
        DatasetRecord(
            dataset_name="mupots-3d",
            release_identity="fixture",
            sample_id=f"TS9/{index:06d}",
            sequence_id="TS9",
            frame_index=index,
            image_path=Path(f"TS9/img_{index:06d}.jpg"),
            image_width=1920,
            image_height=1080,
            people=(),
        )
        for index in range(count)
    ]


def _intrinsics() -> SequenceIntrinsics:
    return SequenceIntrinsics(
        camera=Camera(fx=1100.0, fy=1100.0, cx=960.0, cy=540.0),
        residual_px=0.0,
        samples=len(JOINT_NAMES),
    )


class TestBuildingTheExperiment:
    def test_the_name_carries_the_sequence(self) -> None:
        """The name becomes the artifact directory, so two sequences that shared
        one would overwrite each other's bodies."""
        built = sequence_experiment(_experiment(), sequence_id="TS9", frames_dir=Path("/d/TS9"))

        assert built.name == "mupots-TS9"

    def test_only_the_source_moves(self) -> None:
        base = _experiment()

        built = sequence_experiment(base, sequence_id="TS9", frames_dir=Path("/d/TS9"))

        assert built.ingest.source_path == Path("/d/TS9")
        assert built.ingest.image_fps == base.ingest.image_fps
        assert built.bodies == base.bodies
        assert built.seed == base.seed

    def test_a_frame_directory_without_a_rate_is_refused_here(self) -> None:
        """Ingest would refuse too, but only after the estimator was built."""
        with pytest.raises(GroundTruthError, match="image_fps"):
            sequence_experiment(
                _experiment(image_fps=None), sequence_id="TS9", frames_dir=Path("/d/TS9")
            )


class TestRunningOne:
    def _patch(self, monkeypatch: pytest.MonkeyPatch, frames_total: int) -> list[ExperimentConfig]:
        seen: list[ExperimentConfig] = []

        def fake(
            config: ExperimentConfig,
            *,
            config_dir: Path,
            artifacts_root: Path,
            overwrite: bool = False,
        ) -> BodiesResult:
            seen.append(config)
            return BodiesResult(
                artifact_dir=artifacts_root / config.name,
                artifact=_artifact(frames_total),
                reused=False,
            )

        monkeypatch.setattr(sequences_module, "run_bodies", fake)
        return seen

    def _run(self, *, annotated: int, tmp_path: Path) -> SequenceRun:
        return run_sequence(
            _experiment(),
            sequence_id="TS9",
            frames_dir=tmp_path / "TS9",
            records=_records(annotated),
            intrinsics=_intrinsics(),
            config_dir=tmp_path,
            artifacts_root=tmp_path / "artifacts",
        )

    def test_it_estimates_the_sequence_it_was_given(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = self._patch(monkeypatch, frames_total=4)

        run = self._run(annotated=4, tmp_path=tmp_path)

        assert [config.name for config in seen] == ["mupots-TS9"]
        assert seen[0].ingest.source_path == tmp_path / "TS9"
        assert run.sequence_id == "TS9"

    def test_each_sequence_is_a_separate_call_and_so_a_separate_tracker(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The tracker lives inside the estimator, which `run_bodies` builds. One
        call per sequence is what keeps identities from crossing the boundary."""
        seen = self._patch(monkeypatch, frames_total=4)

        for sequence_id in ("TS9", "TS10"):
            run_sequence(
                _experiment(),
                sequence_id=sequence_id,
                frames_dir=tmp_path / sequence_id,
                records=_records(4),
                intrinsics=_intrinsics(),
                config_dir=tmp_path,
                artifacts_root=tmp_path / "artifacts",
            )

        assert [config.name for config in seen] == ["mupots-TS9", "mupots-TS10"]
        assert len({config.ingest.source_path for config in seen}) == 2

    @pytest.mark.parametrize("annotated", [3, 5])
    def test_a_frame_count_that_disagrees_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, annotated: int
    ) -> None:
        """The one error a later metric cannot see: every frame still gets
        compared against some annotation, just not its own."""
        self._patch(monkeypatch, frames_total=4)

        with pytest.raises(GroundTruthError, match=f"describe {annotated}"):
            self._run(annotated=annotated, tmp_path=tmp_path)

    def test_the_ground_truth_travels_with_the_estimate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A metric that had to go back to disk for the annotations could read a
        different sequence's."""
        self._patch(monkeypatch, frames_total=4)

        run = self._run(annotated=4, tmp_path=tmp_path)

        assert len(run.records) == 4
        assert run.intrinsics.camera.fx == pytest.approx(1100.0)


def test_the_module_does_not_reimplement_inference() -> None:
    """WP2 task 5: every sequence goes through the existing stage [B] path.

    A second estimator here would be a second set of thresholds to keep in step
    with the first, and the numbers would stop being comparable with RQ1-RQ3.
    """
    source = Path(sequences_module.__file__).read_text(encoding="utf-8")

    assert "MultiHmrEstimator" not in source
    assert "run_bodies" in source


class TestReleasingFrames:
    """Stage [1] writes lossless PNG copies of files that are still on disk.

    Releasing them is what lets the benchmark run on a machine with less free
    space than the intermediate. What must be true is that it takes the frames
    and nothing else.
    """

    def _artifact_tree(self, tmp_path: Path) -> tuple[SequenceRun, Path, Path]:
        artifacts_root = tmp_path / "artifacts"
        ingest = artifacts_root / "mupots-TS9" / "ingest" / "0140aa2c6860"
        (ingest / "frames").mkdir(parents=True)
        (ingest / "ingest.json").write_text("{}", encoding="utf-8")
        (ingest / "frames" / "000000.png").write_bytes(b"x" * 4096)
        bodies = artifacts_root / "mupots-TS9" / "bodies" / "abc123"
        bodies.mkdir(parents=True)
        (bodies / "bodies.json").write_text("{}", encoding="utf-8")
        run = SequenceRun(
            sequence_id="TS9",
            experiment_name="mupots-TS9",
            artifact_dir=bodies,
            artifact=_artifact(1),
            records=tuple(_records(1)),
            intrinsics=_intrinsics(),
            reused=False,
        )
        return run, ingest, bodies

    def test_it_frees_the_frames_and_leaves_the_result(self, tmp_path: Path) -> None:
        run, ingest, bodies = self._artifact_tree(tmp_path)

        freed = release_frames(run, artifacts_root=tmp_path / "artifacts")

        assert freed > 4000
        assert not ingest.exists()
        assert (bodies / "bodies.json").is_file()

    def test_releasing_twice_is_not_an_error(self, tmp_path: Path) -> None:
        """The caller asked for the space back, and the space is back."""
        run, _, _ = self._artifact_tree(tmp_path)
        root = tmp_path / "artifacts"

        release_frames(run, artifacts_root=root)

        assert release_frames(run, artifacts_root=root) == 0

    def test_it_touches_nothing_but_the_one_directory_it_was_given(self, tmp_path: Path) -> None:
        """A sibling sequence's frames are somebody else's intermediate."""
        run, _, _ = self._artifact_tree(tmp_path)
        neighbour = tmp_path / "artifacts" / "mupots-TS10" / "ingest" / "beef"
        neighbour.mkdir(parents=True)
        (neighbour / "ingest.json").write_text("{}", encoding="utf-8")

        release_frames(run, artifacts_root=tmp_path / "artifacts")

        assert (neighbour / "ingest.json").is_file()

    def test_a_directory_that_is_not_a_finished_ingest_is_left_alone(self, tmp_path: Path) -> None:
        """Without the descriptor it is a half-written or foreign directory, and
        must not be deleted."""
        run, ingest, _ = self._artifact_tree(tmp_path)
        (ingest / "ingest.json").unlink()

        assert release_frames(run, artifacts_root=tmp_path / "artifacts") == 0
        assert ingest.is_dir()


class TestTellingTheEstimatorTheCamera:
    """The control and the treatment of the focal-length experiment."""

    def test_by_default_the_estimator_guesses(self) -> None:
        built = sequence_experiment(_experiment(), sequence_id="TS9", frames_dir=Path("/d/TS9"))

        assert built.bodies.intrinsics is None
        assert built.name == "mupots-TS9"

    def test_the_annotation_derived_camera_reaches_the_bodies_section(self) -> None:
        camera = annotation_derived_camera(_intrinsics(), width=1920, height=1080)

        built = sequence_experiment(
            _experiment(), sequence_id="TS9", frames_dir=Path("/d/TS9"), camera=camera
        )

        assert built.bodies.intrinsics == camera
        assert built.bodies.intrinsics is not None
        assert built.bodies.intrinsics.fx == pytest.approx(1100.0)
        assert (built.bodies.intrinsics.width, built.bodies.intrinsics.height) == (1920, 1080)

    def test_the_two_conditions_never_share_a_directory(self) -> None:
        """The bodies hash already differs; the name makes it visible on disk."""
        camera = annotation_derived_camera(_intrinsics(), width=1920, height=1080)

        guessed = sequence_experiment(_experiment(), sequence_id="TS9", frames_dir=Path("/d"))
        told = sequence_experiment(
            _experiment(), sequence_id="TS9", frames_dir=Path("/d"), camera=camera
        )

        assert guessed.name != told.name
        assert told.name == "mupots-annotation-derived-camera-TS9"

    def test_an_experiment_that_fixes_its_own_camera_is_refused(self) -> None:
        """One camera for twenty sequences shot with several is the wrong
        experiment, however it was arrived at."""
        base = _experiment().model_copy(
            update={
                "bodies": _experiment().bodies.model_copy(
                    update={
                        "intrinsics": annotation_derived_camera(_intrinsics(), width=1, height=1)
                    }
                )
            }
        )

        with pytest.raises(GroundTruthError, match="per sequence"):
            sequence_experiment(base, sequence_id="TS9", frames_dir=Path("/d"))

    def test_run_sequence_takes_the_image_size_from_the_records(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: list[ExperimentConfig] = []

        def fake(
            config: ExperimentConfig,
            *,
            config_dir: Path,
            artifacts_root: Path,
            overwrite: bool = False,
        ) -> BodiesResult:
            seen.append(config)
            return BodiesResult(
                artifact_dir=artifacts_root / config.name, artifact=_artifact(2), reused=False
            )

        monkeypatch.setattr(sequences_module, "run_bodies", fake)

        run_sequence(
            _experiment(),
            sequence_id="TS9",
            frames_dir=tmp_path / "TS9",
            records=_records(2),
            intrinsics=_intrinsics(),
            config_dir=tmp_path,
            artifacts_root=tmp_path / "artifacts",
            camera="annotation_derived",
        )

        [config] = seen
        assert config.bodies.intrinsics is not None
        assert (config.bodies.intrinsics.width, config.bodies.intrinsics.height) == (1920, 1080)
        assert config.bodies.intrinsics.cx == pytest.approx(960.0)
