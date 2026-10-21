"""Tests for config parsing, validation and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fixture_types import WriteConfig
from unproject.config import (
    BodiesConfig,
    ColmapConfig,
    ConfigError,
    ExperimentConfig,
    IngestConfig,
    KnownIntrinsics,
    LingbotConfig,
    PoseConfig,
    PreprocessConfig,
    _location_fields,
    config_hash,
    load_config,
)

_MINIMAL = """
name = "fixture"

[ingest]
source_path = "clip.avi"
"""


def test_load_minimal_config(write_config: WriteConfig) -> None:
    config = load_config(write_config(_MINIMAL))

    assert config.name == "fixture"
    assert config.seed == 0
    assert config.ingest.source_path == Path("clip.avi")
    assert config.ingest.frame_stride == 1


def test_unknown_key_is_rejected(write_config: WriteConfig) -> None:
    body = _MINIMAL + "frame_strid = 2\n"

    with pytest.raises(ConfigError, match="frame_strid"):
        load_config(write_config(body))


def test_end_time_must_follow_start_time() -> None:
    with pytest.raises(ValueError, match="end_time_seconds"):
        IngestConfig(source_path=Path("clip.avi"), start_time_seconds=5.0, end_time_seconds=2.0)


def test_stride_must_be_positive() -> None:
    with pytest.raises(ValueError, match="frame_stride"):
        IngestConfig(source_path=Path("clip.avi"), frame_stride=0)


def test_unsupported_schema_version_is_rejected(write_config: WriteConfig) -> None:
    body = "schema_version = 99\n" + _MINIMAL

    with pytest.raises(ConfigError, match="schema_version"):
        load_config(write_config(body))


def test_missing_file_reports_the_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(tmp_path / "absent.toml")


def test_invalid_toml_reports_the_path(write_config: WriteConfig) -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(write_config("name = \n"))


def test_hash_ignores_formatting_but_not_values(write_config: WriteConfig) -> None:
    spaced = load_config(write_config(_MINIMAL, name="a.toml"))
    commented = load_config(write_config("# a comment\n" + _MINIMAL, name="b.toml"))
    different = load_config(
        write_config(_MINIMAL + "frame_stride = 3\n", name="c.toml"),
    )

    assert config_hash(spaced.ingest) == config_hash(commented.ingest)
    assert config_hash(spaced.ingest) != config_hash(different.ingest)


def test_hash_is_stable_across_processes() -> None:
    # The hash decides the artifact directory, so it must not drift by accident.
    config = IngestConfig(source_path=Path("clip.avi"))

    assert config_hash(config) == config_hash(IngestConfig(source_path=Path("clip.avi")))
    assert len(config_hash(config)) == 64


def test_experiment_name_must_be_path_safe() -> None:
    with pytest.raises(ValueError, match="name"):
        ExperimentConfig(name="../escape", ingest=IngestConfig(source_path=Path("clip.avi")))


def test_held_out_every_must_leave_something_to_train_on() -> None:
    # 1 would withhold everything, leaving stage [4] with nothing.
    with pytest.raises(ValueError, match="held_out_every"):
        PreprocessConfig(held_out_every=1)


def test_held_out_split_is_on_by_default() -> None:
    # Off by default makes it easy to produce a run that cannot be scored.
    assert PreprocessConfig().held_out_every == 8


def test_zero_disables_an_optional_integer() -> None:
    # TOML has no null literal, so an option with a non-null default needs a
    # sentinel to be switchable off from a config file at all.
    config = PreprocessConfig.model_validate({"held_out_every": 0, "max_long_edge": 0})

    assert config.held_out_every is None
    assert config.max_long_edge is None


def test_the_pose_backend_defaults_to_colmap() -> None:
    # The reference baseline, so a config that says nothing gets the one that works.
    assert PoseConfig().backend == "colmap"


def test_an_unknown_pose_backend_is_rejected() -> None:
    with pytest.raises(ValueError):
        PoseConfig.model_validate({"backend": "mystery"})


def test_the_colmap_matcher_choices_are_closed() -> None:
    with pytest.raises(ValueError):
        ColmapConfig.model_validate({"matcher": "approximate"})


def test_where_the_weights_live_is_not_part_of_a_runs_identity() -> None:
    # An artifact is keyed by the run that would produce it. Two configs that
    # differ only in which directory the weights were cached in describe the
    # same run, and re-deriving nine minutes of COLMAP because a cache moved is
    # a bug rather than caution.
    here = LingbotConfig()
    elsewhere = LingbotConfig(cache_dir=Path("/somewhere/else"))
    assert here.cache_dir != elsewhere.cache_dir
    assert config_hash(here) == config_hash(elsewhere)


def test_the_exclusion_reaches_into_nested_sections() -> None:
    # Pydantic's own `exclude` does not consult a nested model's settings, so
    # this is the case that actually broke: a pose config holds the backend
    # config, and hashing the outer one used to drag the cache path in with it.
    assert config_hash(PoseConfig()) == config_hash(
        PoseConfig(lingbot=LingbotConfig(cache_dir=Path("/somewhere/else")))
    )


def test_a_real_setting_still_changes_the_identity() -> None:
    # The exclusion has to be narrow, or it would quietly stop distinguishing
    # runs that really are different.
    assert config_hash(LingbotConfig()) != config_hash(LingbotConfig(kv_cache_sliding_window=32))
    assert config_hash(LingbotConfig()) != config_hash(LingbotConfig(sha256="0" * 64))


# --- a camera the estimator is told -------------------------------------------

# The hash of a default BodiesConfig on the day `intrinsics` was added. Every
# bodies artifact produced before that day is addressed by it; if this number
# moves, all of them stop matching their own configuration.
_BODIES_HASH_BEFORE_INTRINSICS = "d1d843587d67c93db6d3c0db12d76d1781efdbe856073b7aed42351466aab6c9"


def _camera() -> KnownIntrinsics:
    return KnownIntrinsics(width=1920, height=1080, fx=1100.0, fy=1100.0, cx=960.0, cy=540.0)


def test_an_absent_camera_hashes_exactly_as_before_the_field_existed() -> None:
    assert config_hash(BodiesConfig()) == _BODIES_HASH_BEFORE_INTRINSICS


def test_a_known_camera_is_a_different_run() -> None:
    """Bodies placed with a known focal length are not the bodies placed with a
    guessed one, so they must not share an artifact."""
    assert config_hash(BodiesConfig(intrinsics=_camera())) != config_hash(BodiesConfig())


def test_two_cameras_are_two_runs() -> None:
    longer = _camera().model_copy(update={"fx": 1300.0, "fy": 1300.0})

    assert config_hash(BodiesConfig(intrinsics=_camera())) != config_hash(
        BodiesConfig(intrinsics=longer)
    )


def test_the_absent_camera_is_also_absent_from_a_nested_hash() -> None:
    """The experiment holds the bodies section; the rule has to reach it there."""
    experiment = ExperimentConfig.model_validate(
        {"name": "x", "ingest": {"source_path": "clip.avi"}}
    )

    assert '"intrinsics"' not in experiment.model_dump_json(exclude=_location_fields(experiment))


@pytest.mark.parametrize("field", ["width", "height", "fx", "fy"])
def test_a_camera_with_a_non_positive_dimension_is_refused(field: str) -> None:
    with pytest.raises(ValidationError):
        KnownIntrinsics.model_validate({**_camera().model_dump(), field: 0})


_PROVENANCE = """
name = "pedestrians"
[ingest]
source_path = "clip.mp4"
[provenance]
title = "Video Codec Test: pedestrian area"
author = "a contributor"
source_url = "https://commons.wikimedia.org/wiki/File:x.webm"
licence = "CC0 1.0"
acquired_on = 2026-09-06
original_sha256 = "bfadaa62cccb42db875d50bb842aa0964fbf72040432e4097c1df59e043e0c26"
original_bytes = 11215394
derivation = "ffmpeg -i x.webm clip.mp4"
"""


def test_provenance_is_read_when_declared(write_config: WriteConfig) -> None:
    config = load_config(write_config(_PROVENANCE))

    assert config.provenance is not None
    assert config.provenance.licence == "CC0 1.0"
    assert config.provenance.acquired_on.isoformat() == "2026-09-06"


def test_provenance_is_absent_not_invented(write_config: WriteConfig) -> None:
    config = load_config(write_config('name = "x"\n[ingest]\nsource_path = "c.mp4"\n'))

    assert config.provenance is None


def test_a_malformed_digest_in_provenance_is_refused(write_config: WriteConfig) -> None:
    with pytest.raises(ConfigError, match="original_sha256"):
        load_config(write_config(_PROVENANCE.replace("bfadaa62", "not-hex-")))


def test_provenance_does_not_enter_any_stage_identity(write_config: WriteConfig) -> None:
    """Who made the footage does not change what a stage computes from it."""
    with_it = load_config(write_config(_PROVENANCE, name="a.toml"))
    without = load_config(
        write_config('name = "pedestrians"\n[ingest]\nsource_path = "clip.mp4"\n', name="b.toml")
    )

    assert config_hash(with_it) == config_hash(without)
    assert config_hash(with_it.ingest) == config_hash(without.ingest)
    assert config_hash(with_it.bodies) == config_hash(without.bodies)
