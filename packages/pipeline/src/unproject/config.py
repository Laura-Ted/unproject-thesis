"""Experiment configuration: the TOML files under `experiments/`.

Nothing here touches the filesystem beyond reading the config file itself, so
`validate-config` also works on a machine with no datasets checked out.
"""

from __future__ import annotations

import hashlib
import tomllib
from datetime import date
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1

# What `model_dump_json(exclude=...)` accepts: a field is either dropped whole or
# has its own nested map of what to drop.
type Exclusions = dict[str, bool | "Exclusions"]


class ConfigError(Exception):
    """Raised when a config file is unreadable or fails validation."""


class _StrictModel(BaseModel):
    # extra="forbid": a typo in a key must fail, not fall back to the default.
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Fields that say where to find something rather than what to compute.
    # `config_hash` leaves them out: an artifact is keyed by the run that would
    # produce it, and moving a file does not make it a different run.
    LOCATION_FIELDS: ClassVar[frozenset[str]] = frozenset()


def _zero_means_off(value: object) -> object:
    """Read 0 as "this feature is off".

    TOML has no null literal, and omitting the key cannot serve either since
    these options have non-null defaults.
    """
    return None if value == 0 else value


class KnownIntrinsics(_StrictModel):
    """A camera the operator knows, handed to the estimator instead of guessed.

    Multi-HMR predicts its own field of view and converts everything it sees to
    metres through that one number: in the decoder, distance is the focal length
    divided by an apparent size. So a wrong guess is a wrong depth by the same
    factor, and a known camera removes that factor at the source.

    Pixels at the resolution of the frames the estimator is given. The size is
    part of the record because a focal length is only meaningful at one.
    """

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0.0)
    fy: float = Field(gt=0.0)
    cx: float
    cy: float


class FrameGeometry(_StrictModel):
    """A geometry applied to every frame as the ingest writes it (ADR-0006 §5).

    Both operations are exact integer arithmetic on the decoded pixels, so a
    machine that reads the same source bytes writes the same frames: the
    matrix suite's four variants of one sequence derive from the official
    archive on every machine, with no transcode to be carried or trusted.
    `downscale` is a box average over factor x factor blocks (the frame size
    must divide); `letterbox_square` then pads with black bands, centred, to
    a square of the long edge.
    """

    downscale: int = Field(default=1, ge=1)
    letterbox_square: bool = False

    @model_validator(mode="after")
    def _not_the_identity(self) -> Self:
        if self.downscale == 1 and not self.letterbox_square:
            msg = "geometry does nothing; omit the table instead of declaring the identity"
            raise ValueError(msg)
        return self


class IngestConfig(_StrictModel):
    """Stage [1] parameters."""

    # Absent from the hash while unset, so every ingest config written before
    # the field existed keeps the identity of the artifacts it produced.
    HASH_OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"geometry"})

    # A video file or a directory of extracted frames; relative paths resolve
    # against the config file's own directory.
    source_path: Path
    # Applied to every written frame; None writes the frames as decoded.
    geometry: FrameGeometry | None = None
    # Required when `source_path` is a directory: images carry no frame rate, so
    # one has to be declared for timestamps to mean anything.
    image_fps: float | None = Field(default=None, gt=0.0)
    # 1 keeps every frame. Content-aware selection comes later, in stage [2].
    frame_stride: int = Field(default=1, ge=1)
    start_time_seconds: float | None = Field(default=None, ge=0.0)
    end_time_seconds: float | None = Field(default=None, gt=0.0)
    max_frames: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_time_window(self) -> Self:
        start = self.start_time_seconds
        end = self.end_time_seconds
        if start is not None and end is not None and end <= start:
            msg = f"end_time_seconds ({end}) must be greater than start_time_seconds ({start})"
            raise ValueError(msg)
        return self


class PreprocessConfig(_StrictModel):
    """Stage [2] parameters."""

    # Caps the long edge of the written keyframes; 0 keeps the input size.
    # Never enlarges, which would only claim a resolution that is not there.
    max_long_edge: int | None = Field(default=1600, gt=0)

    _off_if_zero = field_validator("max_long_edge", "held_out_every", mode="before")(
        _zero_means_off
    )

    # Minimum motion between keyframes, in output-image pixels. Below this the
    # view rays are near parallel and triangulation is ill-conditioned.
    min_disparity_px: float = Field(default=12.0, gt=0.0)
    # Fraction of tracked points that must survive. A collapse means the pair is
    # already too far apart to match.
    min_tracked_ratio: float = Field(default=0.3, gt=0.0, le=1.0)
    # A percentile of the sequence, not an absolute value: sharpness depends on
    # how much texture a scene has.
    blur_percentile: float = Field(default=20.0, ge=0.0, lt=100.0)
    # A blurred stretch that produces nothing leaves a hole, and the chain of
    # keyframes breaks in two.
    max_skipped_frames: int = Field(default=30, ge=1)
    # A handheld take often starts while the shot is still being framed.
    first_frame_window: int = Field(default=10, ge=1)
    # Fixed rather than native, so the metrics compare across scenes shot at
    # different resolutions.
    metric_long_edge: int = Field(default=480, gt=0)

    # Every Nth keyframe is withheld from training and scored at stage [5]. 8
    # follows Mip-NeRF 360 and Kerbl et al. so the numbers sit alongside the
    # published ones. Changing it after results exist invalidates them, so it is
    # fixed once rather than tuned.
    held_out_every: int | None = Field(default=8, ge=2)

    # Undistortion needs intrinsics and stage [3] is what produces them, so this
    # stays unset until there is a calibration file to test against.
    calibration_path: Path | None = None

    @model_validator(mode="after")
    def _check_undistortion_unsupported(self) -> Self:
        if self.calibration_path is not None:
            msg = (
                "calibration_path is not supported yet: undistortion needs camera "
                "intrinsics, and stage [3] is what estimates them"
            )
            raise ValueError(msg)
        return self


class ColmapConfig(_StrictModel):
    """Reference SfM backend parameters."""

    # "auto" is sequential for video and exhaustive for an image directory,
    # which promises no ordering.
    matcher: Literal["auto", "sequential", "exhaustive"] = "auto"
    # COLMAP's default for an uncalibrated camera: one radial term estimates
    # more stably than the full model.
    camera_model: str = "SIMPLE_RADIAL"
    # A handheld take is one physical camera, so share the intrinsics rather
    # than solving them per image.
    single_camera: bool = True


class LingbotConfig(_StrictModel):
    """Learned backend parameters.

    Weights are pinned by revision and checksum, so an upstream reupload cannot
    change what a run produces.
    """

    repo_id: str = "robbyant/lingbot-map"
    checkpoint: str = "lingbot-map.pt"
    revision: str = "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9"
    sha256: str = "ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72"

    # Resolved relative to the config file, like `source_path`. Inside the repo
    # rather than a user-wide cache, and gitignored. Not part of the run's
    # identity: `revision` and `sha256` already say which weights.
    cache_dir: Path = Path("../models")

    LOCATION_FIELDS: ClassVar[frozenset[str]] = frozenset({"cache_dir"})

    # What the model was trained at. Images are resized and centre-cropped to
    # this, and the intrinsics mapped back afterwards.
    image_size: int = Field(default=518, gt=0)

    # How many past frames the KV cache keeps, and the setting that decides
    # whether a run fits in memory. Upstream's 64 peaks near 16 GB and grows
    # with the sequence; 16 holds flat around 10 GB.
    kv_cache_sliding_window: int = Field(default=16, gt=0)
    # Frames attended to bidirectionally before streaming starts. Upstream uses
    # 8; 2 costs nothing measurable and lowers the peak.
    num_scale_frames: int = Field(default=2, gt=0)
    # Refinement passes in the camera head. Fewer is faster and less accurate.
    camera_num_iterations: int = Field(default=4, gt=0)
    # bfloat16 halves the trunk and is what makes the run fit; float32 is kept
    # to measure what that costs in accuracy.
    dtype: Literal["bfloat16", "float32"] = "bfloat16"

    # The backend predicts dense depth, so the cloud is back-projected and then
    # thinned; left dense it would be tens of millions of points.
    max_points: int = Field(default=200_000, gt=0)
    # A percentile rather than an absolute value, since the head's confidence is
    # unitless.
    depth_confidence_percentile: float = Field(default=50.0, ge=0.0, lt=100.0)


class PoseConfig(_StrictModel):
    """Stage [3] parameters.

    The backend is the experiment's variable; everything else is held constant.
    """

    backend: Literal["colmap", "lingbot"] = "colmap"
    colmap: ColmapConfig = ColmapConfig()
    lingbot: LingbotConfig = LingbotConfig()


class BodiesConfig(_StrictModel):
    """Stage [B] parameters.

    One estimator, so no backend discriminator; a second one would earn the
    field. Weights are pinned by digest rather than by name, so an upstream
    reupload cannot quietly change what a run produces.
    """

    checkpoint_url: str = (
        "https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multihmr2.pt"
    )
    sha256: str = "fd2d9ab7010a5f590a3db2480f5f476fd3ec6afbbde80babbec0d821ba9763d6"

    # Resolved relative to the config file, like `source_path`. Not part of the
    # run's identity: `sha256` says which weights, this only says where they sit.
    cache_dir: Path = Path("../models")

    LOCATION_FIELDS: ClassVar[frozenset[str]] = frozenset({"cache_dir"})

    # Added after artifacts existed. Absent means the estimator guessed its
    # camera, which is what every earlier run did, so an absent value has to
    # hash exactly as those runs did or all of them would be re-derived.
    HASH_OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"intrinsics"})

    # When set, the estimator is told the camera rather than left to guess it.
    # Part of the run's identity: bodies placed with a known focal length are
    # not the same result as bodies placed with an invented one.
    intrinsics: KnownIntrinsics | None = None

    # The backbone is fetched as source by torch.hub at model-build time, so a
    # branch name here would leave the run irreproducible even with the
    # checkpoint fixed.
    backbone_repo: str = "facebookresearch/dinov3"
    backbone_ref: str = "6876159a11b4df116f30f667f8c9888617df0751"

    # A body prior returns a plausible human wherever it looks, so this
    # threshold is the only filter against false detections.
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    # Detections whose pelvises land closer than this are treated as one person.
    nms_distance_metres: float = Field(default=0.25, gt=0.0)


class Provenance(_StrictModel):
    """Where a scene's footage came from, and on what terms.

    ADR-0005 WP2 requires a benchmark input to carry source, author, licence,
    URL, acquisition date and digest. Recorded on the experiment rather than
    the benchmark because it belongs to the footage, whatever measures it.

    `original_sha256` is the digest of the file as obtained from the source,
    before any transcode; the ingest artifact records the digest of what was
    actually decoded, and `derivation` is the command that turned one into the
    other. Two digests, because a transcode is not bit-stable across encoder
    versions and the one that is checkable against the publisher is the
    original's.
    """

    title: str = Field(min_length=1)
    author: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    acquired_on: date
    original_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_bytes: int = Field(gt=0)
    # The exact command(s) from the original to the file `source_path` names.
    derivation: str = Field(min_length=1)
    # Where the bytes of the original can be fetched from, when that is a file
    # rather than a page; `source_url` stays the page that states the licence.
    download_url: str | None = None
    # A snapshot of `source_url` in a web archive, taken when the footage was
    # acquired, so the licence statement outlives the page (ADR-0006 §5).
    archived_url: str | None = None
    # A citable deposit of the footage itself - a DOI - where the licence
    # allows one to be made; None where it does not.
    deposit_doi: str | None = None
    # The digest of what `source_path` names, as the ingest computes it: the
    # file's, for a video; the ordered content digest, for an image directory.
    # Set it once the campaign's input is fixed, and every machine's preflight
    # refuses to measure different bytes under the same name.
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ExperimentConfig(_StrictModel):
    """A run's inputs. Later stages add their own sections here."""

    # Descriptive, not computational: who made the footage does not change what
    # a stage computes from it, and no stage hashes the whole experiment anyway.
    # Listed so the intent survives if one ever does.
    LOCATION_FIELDS: ClassVar[frozenset[str]] = frozenset({"provenance"})

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    # Also the artifact directory name, hence the character restriction.
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    description: str = ""
    # Recorded in every stage's run manifest; no stage computes with it, since
    # the held-out split is positional (`held_out_every`) and needs no seed.
    seed: int = 0
    ingest: IngestConfig
    preprocess: PreprocessConfig = PreprocessConfig()
    pose: PoseConfig = PoseConfig()
    # A sibling of [2] rather than a successor: it reads ingest directly. Each
    # stage chains only from its own section, so adding this leaves the others
    # hashing to what they did before.
    bodies: BodiesConfig = BodiesConfig()
    # Absent for footage whose origin was never recorded, which is itself a
    # fact worth being able to read off the file.
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != SCHEMA_VERSION:
            msg = (
                f"config schema_version {self.schema_version} is not supported "
                f"by this build (expected {SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self


def load_config(path: Path) -> ExperimentConfig:
    """Parse and validate a TOML experiment config."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error

    try:
        return ExperimentConfig.model_validate(data)
    except ValueError as error:
        raise ConfigError(f"{path} failed validation:\n{error}") from error


def config_hash(config: BaseModel) -> str:
    """Content hash of a validated config section.

    Hashes the model rather than the file bytes, so comments and key order do not
    affect it; a changed default does, which is intended.

    Fields listed in `LOCATION_FIELDS` are left out: they say where something is,
    not what to compute, and re-deriving an artifact because a cache moved is a
    bug rather than caution.
    """
    excluded = _location_fields(config)
    canonical = config.model_dump_json(exclude=excluded or None)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _location_fields(model: BaseModel) -> Exclusions:
    """Collect `LOCATION_FIELDS` from a model and everything nested inside it.

    Pydantic's `exclude` does not consult a nested model's settings, so a section
    holding another has to name the nested exclusions itself.

    `HASH_OPTIONAL_FIELDS` ride along when they hold nothing: the dump would
    otherwise carry `"field": null`, and every artifact produced before the
    field existed would stop matching its own configuration.
    """
    excluded: Exclusions = dict.fromkeys(getattr(model, "LOCATION_FIELDS", frozenset()), True)
    for name in getattr(model, "HASH_OPTIONAL_FIELDS", frozenset()):
        if getattr(model, name) is None:
            excluded[name] = True
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel) and (nested := _location_fields(value)):
            excluded[name] = nested
    return excluded
