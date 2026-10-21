"""Tests for stage [B]'s identity and its place in the pipeline.

Nothing here runs the estimator. Getting artifact identity wrong either reuses
the wrong result or throws away work that was still good.
"""

from __future__ import annotations

from pathlib import Path

from unproject.bodies.stage import bodies_artifact_dir, bodies_hash
from unproject.config import BodiesConfig, ExperimentConfig, IngestConfig, config_hash
from unproject.preprocess.stage import preprocess_hash


def _config(**overrides: object) -> ExperimentConfig:
    defaults: dict[str, object] = {
        "name": "crowd",
        "ingest": IngestConfig(source_path=Path("../datasets/crowd/clip.mp4"), frame_stride=2),
    }
    return ExperimentConfig.model_validate(defaults | overrides)


def test_the_same_config_lands_in_the_same_place() -> None:
    assert bodies_hash("abc", BodiesConfig()) == bodies_hash("abc", BodiesConfig())


def test_different_frames_mean_different_bodies() -> None:
    """Bodies fitted to one set of frames must not be reused for another."""
    assert bodies_hash("abc", BodiesConfig()) != bodies_hash("def", BodiesConfig())


def test_a_changed_threshold_changes_the_identity() -> None:
    # It decides which detections reach the artifact, so it is part of the run.
    assert bodies_hash("abc", BodiesConfig()) != bodies_hash(
        "abc", BodiesConfig(confidence_threshold=0.6)
    )


def test_a_changed_backbone_ref_changes_the_identity() -> None:
    # The backbone is executed code fetched at build time; a different commit is
    # a different run even with the same weights.
    assert bodies_hash("abc", BodiesConfig()) != bodies_hash(
        "abc", BodiesConfig(backbone_ref="0" * 40)
    )


def test_moving_the_weights_does_not_change_the_identity() -> None:
    """`cache_dir` says where the weights sit, not what the run computes.

    Letting it into the hash orphans every artifact the moment someone points it
    somewhere else.
    """
    moved = BodiesConfig(cache_dir=Path("/somewhere/else"))
    assert bodies_hash("abc", BodiesConfig()) == bodies_hash("abc", moved)
    assert config_hash(BodiesConfig()) == config_hash(moved)


def test_adding_the_bodies_section_did_not_orphan_the_other_stages() -> None:
    """A sibling stage must not disturb the hashes upstream of it.

    If this fails, every artifact already on disk has been stranded.
    """
    config = _config()
    ingest_hash = config_hash(config.ingest)
    # The values these produced before stage [B] was added, from the artifact
    # directories on disk for the crowd experiment.
    assert ingest_hash.startswith("0140aa2c6860")
    assert preprocess_hash(ingest_hash[:12], config.preprocess).startswith("a4086014d25c")


def test_the_artifact_directory_is_keyed_by_the_chained_hash() -> None:
    chained = bodies_hash("abc", BodiesConfig())
    directory = bodies_artifact_dir(Path("artifacts"), "crowd", chained)
    assert directory == Path("artifacts/crowd/bodies") / chained[:12]


def test_bodies_is_reachable_from_a_config_that_never_mentions_it() -> None:
    """An existing experiment file gains the stage without being edited."""
    config = _config()
    assert config.bodies == BodiesConfig()
