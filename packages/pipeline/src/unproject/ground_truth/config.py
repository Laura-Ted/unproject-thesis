"""What one ground-truth evaluation is, as a versioned file.

Separate from the RQ1/RQ2 benchmark schema on purpose: they measure different
things and sharing a schema would let a change to one silently alter the other.

Paths in here say *where* something is and never contribute to identity; see
`identity_hash`.

**Version 2 renames, and nothing else.** Version 1 called the greedy 40-pixel
association `official`, the annotation-derived camera `released`, and reported
a project metric under the official protocol's name. ADR-0005 §3 reserves those
words for things this project does not do, so every value here now says what
it is: a matcher named by its algorithm and radius, a camera named by where it
came from, and a metric protocol identifier that a table can be traced back to.
A v1 file is refused with the rename table rather than read under the old
names; a v1 *artifact* is still read, through `upgrade_config_data`.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.ground_truth.dataset import GroundTruthError

GROUND_TRUTH_SCHEMA_VERSION = 2

type DatasetKind = Literal["mupots-3d", "anny-one"]

# Which person-association algorithm the result was computed with. Part of
# the config, and so of the evaluation identity, because two runs that paired
# people differently are two different measurements and must not share an
# output directory.
#
# `mupots_greedy_2d_40px` reproduces the released evaluator's association: for
# each annotated person, in annotation order, the unclaimed prediction with the
# most joints within 40 px on each axis.
#
# `normalized_global_assignment` is minimum-cost bipartite assignment on 2D
# distance normalised by the image diagonal, gated by `association_gate`.
type MatcherKind = Literal["mupots_greedy_2d_40px", "normalized_global_assignment"]

# Where the estimator's camera comes from. `model_estimated` is the model's own
# guess and the ordinary condition. `annotation_derived` hands it the focal
# length and principal point recovered by least squares from the released 2D
# and 3D annotations: not a published calibration, which MuPoTS does not ship,
# and never to be called a released or true camera. Two runs differing only
# here are the control and the treatment of the focal-length experiment.
type CameraKind = Literal["model_estimated", "annotation_derived"]

# Which definition of the metrics a result was computed under. One identifier
# per definition; a change to joints, scale field, association or aggregation
# is a new identifier, never a silent edit to an existing one.
type MetricProtocol = Literal["unproject_mupots_14j_v1"]

# What is done to the frames before the estimator sees them. `none` is the
# release as shipped. `letterbox_square` pads every wide sequence with black
# rows to a square and moves the 2D annotations and the derived principal
# point down by the padding: the same content in a square tensor, which is
# the controlled test of the aspect-ratio effect ADR-0005 WP5 asks for. Part
# of the identity: transformed frames are a different input.
type FrameTransform = Literal["none", "letterbox_square"]

# The v1 spellings, so a v1 artifact can still be read and a v1 config file can
# be told exactly what to change.
LEGACY_MATCHERS: dict[str, MatcherKind] = {
    "official": "mupots_greedy_2d_40px",
    "native": "normalized_global_assignment",
}
LEGACY_CAMERAS: dict[str, CameraKind] = {
    "estimated": "model_estimated",
    "released": "annotation_derived",
}

# The released evaluator's radius, per axis, in pixels. It is part of the
# matcher's name rather than a threshold because changing it would be a
# different algorithm, not a different setting of this one.
MUPOTS_GREEDY_RADIUS_PX = 40.0


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    """What a metric protocol identifier promises, so a table can say it."""

    identifier: MetricProtocol
    # Joints of the common skeleton, root included.
    common_joints: int
    root_joint: str
    # How many of the 14 official PCK joints the common skeleton covers, and
    # which official joint it lacks. The official evaluator uses all 14.
    official_pck_joints_covered: int
    official_pck_joints_total: int
    absent_official_joints: tuple[str, ...]
    # Which released field the root-relative metrics are taken on. The official
    # evaluator uses `univ_annot3`; this protocol uses `annot3`.
    root_relative_field: str
    pck_radius_mm: float
    description: str


PROTOCOLS: dict[MetricProtocol, ProtocolSpec] = {
    "unproject_mupots_14j_v1": ProtocolSpec(
        identifier="unproject_mupots_14j_v1",
        common_joints=14,
        root_joint="pelvis",
        official_pck_joints_covered=13,
        official_pck_joints_total=14,
        absent_official_joints=("head_top",),
        root_relative_field="annot3",
        pck_radius_mm=150.0,
        description=(
            "Unproject protocol on MuPoTS-3D: MuPoTS-compatible project metric over 14 "
            "common joints (pelvis plus 13 of the 14 official PCK joints; head_top has no "
            "Anny counterpart), root-relative error taken on annot3 rather than "
            "univ_annot3, and micro/macro aggregation over matched people. Not the "
            "official evaluator's score."
        ),
    ),
}


class MetricThresholds(BaseModel):
    """Every number a metric compares against, frozen before results are seen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The official MuPoTS radius. Millimetres, because that is what the protocol
    # states and converting it here would invite a silent factor of a thousand.
    pck_radius_mm: float = Field(default=150.0, gt=0.0)
    # Only the global matcher reads this: association is 2D and normalised by
    # the image diagonal, so it is a fraction rather than a pixel count and does
    # not change with resolution. The MuPoTS matcher has its own fixed radius
    # and refuses a value here, so a number that does nothing cannot be
    # recorded as if it did.
    association_gate: float | None = Field(default=None, gt=0.0, le=1.0)
    # Ground-truth distance bins for the depth breakdown, in metres, ascending.
    depth_bins_metres: list[float] = Field(default=[2.0, 4.0, 6.0, 8.0])
    # A velocity across a longer interval describes the gap, not the person.
    max_temporal_interval_seconds: float = Field(default=0.2, gt=0.0)

    @model_validator(mode="after")
    def _check_bins_ascend(self) -> Self:
        if self.depth_bins_metres != sorted(self.depth_bins_metres):
            msg = f"depth_bins_metres must ascend, got {self.depth_bins_metres}"
            raise ValueError(msg)
        if len(set(self.depth_bins_metres)) != len(self.depth_bins_metres):
            msg = f"depth_bins_metres repeats an edge: {self.depth_bins_metres}"
            raise ValueError(msg)
        return self


class GroundTruthConfig(BaseModel):
    """One evaluation of one release."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Fields that say where something is rather than what to compute. Excluded
    # from identity, exactly as `BodiesConfig.cache_dir` is.
    LOCATION_FIELDS: ClassVar[frozenset[str]] = frozenset({"dataset_root", "output_root"})

    schema_version: int = Field(default=GROUND_TRUTH_SCHEMA_VERSION, ge=1)
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    description: str = ""

    dataset: DatasetKind
    dataset_root: Path
    # What the author actually downloaded. Free text because no release here
    # publishes a version string in a fixed form; the inventory hash is what
    # actually pins the content.
    release: str = Field(min_length=1)
    split: str = Field(default="test", min_length=1)
    # Required for Anny-One, where a subset is mandatory. See ADR-0004 §4.4.
    subset_manifest: Path | None = None

    # The experiment supplying the estimator settings and the pinned checkpoint.
    experiment: Path
    mapping: Path

    metric_protocol: MetricProtocol = "unproject_mupots_14j_v1"
    matcher: MatcherKind = "mupots_greedy_2d_40px"
    camera: CameraKind = "model_estimated"
    frame_transform: FrameTransform = "none"
    thresholds: MetricThresholds = MetricThresholds()
    output_root: Path = Path("../artifacts/ground-truth")
    seed: int = 0
    # Set when the official evaluator is used, pinned by revision or checksum.
    official_evaluator: str | None = None

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != GROUND_TRUTH_SCHEMA_VERSION:
            msg = (
                f"ground-truth schema_version {self.schema_version} is not supported "
                f"by this build (expected {GROUND_TRUTH_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_the_gate_belongs_to_the_matcher(self) -> Self:
        """The normalised gate is the global matcher's parameter and nobody else's.

        Refused under the MuPoTS matcher and required under the global one; the
        reason is on `MetricThresholds.association_gate`.
        """
        gate = self.thresholds.association_gate
        if self.matcher == "mupots_greedy_2d_40px" and gate is not None:
            msg = (
                f"association_gate = {gate} does nothing under matcher "
                f"'mupots_greedy_2d_40px', whose {MUPOTS_GREEDY_RADIUS_PX:.0f} px per-axis "
                "radius is fixed by its name. Remove it, or choose "
                "'normalized_global_assignment', which is what reads it."
            )
            raise ValueError(msg)
        if self.matcher == "normalized_global_assignment" and gate is None:
            msg = (
                "matcher 'normalized_global_assignment' needs thresholds.association_gate, "
                "a fraction of the image diagonal above which a pair may not match"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_anny_one_declares_a_subset(self) -> Self:
        """Anny-One is 780,964 images across 122 GB. Evaluating it whole is not a
        default anyone should reach by omission."""
        if self.dataset == "anny-one" and self.subset_manifest is None:
            msg = (
                "anny-one requires subset_manifest: the release is 780,964 images "
                "and ADR-0004 §4.4 forbids evaluating it whole. Generate a "
                "deterministic subset and commit it before running inference."
            )
            raise ValueError(msg)
        return self

    @property
    def protocol(self) -> ProtocolSpec:
        return PROTOCOLS[self.metric_protocol]


def upgrade_config_data(data: dict[str, object]) -> dict[str, object]:
    """Rewrite a schema-1 config mapping into schema 2. Labels only.

    Used for the config embedded in a v1 evaluation descriptor, so an old
    result stays readable under the names it should have had. The numbers it
    describes are untouched: the same matcher under a new name pairs the same
    people. A gate recorded beside the greedy matcher is dropped, because it
    never did anything there, and that is the one field whose absence a reader
    might notice - it is noted, not hidden, in the descriptor's migration
    record.
    """
    version = data.get("schema_version", 1)
    if version != 1:
        raise GroundTruthError(f"cannot upgrade a schema_version {version} ground-truth config")

    upgraded = dict(data)
    upgraded["schema_version"] = GROUND_TRUTH_SCHEMA_VERSION
    matcher = str(upgraded.get("matcher", "official"))
    camera = str(upgraded.get("camera", "estimated"))
    if matcher not in LEGACY_MATCHERS:
        raise GroundTruthError(f"schema 1 has no matcher {matcher!r}")
    if camera not in LEGACY_CAMERAS:
        raise GroundTruthError(f"schema 1 has no camera {camera!r}")
    upgraded["matcher"] = LEGACY_MATCHERS[matcher]
    upgraded["camera"] = LEGACY_CAMERAS[camera]
    upgraded.setdefault("metric_protocol", "unproject_mupots_14j_v1")

    recorded = upgraded.get("thresholds")
    if recorded is not None and not isinstance(recorded, dict):
        raise GroundTruthError(f"thresholds is {type(recorded).__name__}, not a table")
    thresholds: dict[str, object] = dict(recorded) if isinstance(recorded, dict) else {}
    if upgraded["matcher"] == "mupots_greedy_2d_40px":
        thresholds.pop("association_gate", None)
    else:
        # v1 defaulted the gate to 0.1 when absent; make the value explicit.
        thresholds.setdefault("association_gate", 0.1)
    upgraded["thresholds"] = thresholds
    return upgraded


def _rename_table() -> str:
    lines = ["schema_version = 1  ->  schema_version = 2"]
    lines.extend(
        f'matcher = "{old}"  ->  matcher = "{new}"' for old, new in LEGACY_MATCHERS.items()
    )
    lines.extend(f'camera = "{old}"  ->  camera = "{new}"' for old, new in LEGACY_CAMERAS.items())
    lines.append('metric_protocol = "unproject_mupots_14j_v1"  (new, declare it)')
    lines.append(
        "thresholds.association_gate: keep only with normalized_global_assignment, "
        "which is the matcher that reads it"
    )
    return "\n  ".join(lines)


def load_ground_truth_config(path: Path) -> GroundTruthConfig:
    """Parse and validate a ground-truth evaluation TOML."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise GroundTruthError(f"cannot read {path}: {error}") from error

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise GroundTruthError(f"{path} is not valid TOML: {error}") from error

    if data.get("schema_version", GROUND_TRUTH_SCHEMA_VERSION) == 1:
        raise GroundTruthError(
            f"{path} is a schema 1 ground-truth config. Schema 2 renames its values so "
            "they say what they are (ADR-0005 WP1); edit the file:\n  " + _rename_table()
        )

    try:
        return GroundTruthConfig.model_validate(data)
    except ValueError as error:
        raise GroundTruthError(f"{path} failed validation:\n{error}") from error


def identity_hash(config: GroundTruthConfig, *, inventory_hash: str, mapping_hash: str) -> str:
    """What this evaluation is, independent of where its inputs sit.

    Location fields are dropped and content hashes take their place. Moving the
    dataset to another disk reuses the result; changing a byte inside it does
    not, which is the whole distinction ADR-0002 §7.2 is about.
    """
    canonical = config.model_dump_json(
        exclude=dict.fromkeys(GroundTruthConfig.LOCATION_FIELDS, True)
    )
    combined = f"{canonical}:{inventory_hash}:{mapping_hash}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def resolve(path: Path, base: Path) -> Path:
    """Relative paths resolve against the config file, as everywhere else here."""
    return path if path.is_absolute() else (base / path).resolve()
