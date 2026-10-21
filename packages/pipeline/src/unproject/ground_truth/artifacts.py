"""What an evaluation leaves behind, and what makes it trustworthy.

`evaluation.json` is the descriptor and the completion marker. It is written
last, into a directory that was built somewhere else, so a reader finds either a
finished evaluation or nothing claiming to be one.

Every file it references carries a byte size and a SHA-256. A descriptor that
names a sidecar it cannot vouch for is worse than no descriptor: it looks
complete.

Version 2 of the descriptor says three things version 1 left to the reader: the
metric protocol the numbers were computed under, the matcher and the one
parameter it actually used, and where each body artifact's inference came from
- commit and dirty state - so a score traces to the code that produced the
bodies and not only to the code that scored them. A version 1 file is read by
upgrading its labels in memory; the numbers it points at are not touched, and
the upgrade is recorded in the object so nobody mistakes it for a native v2.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.ground_truth.config import (
    MUPOTS_GREEDY_RADIUS_PX,
    CameraKind,
    GroundTruthConfig,
    MatcherKind,
    MetricProtocol,
    upgrade_config_data,
)
from unproject.ground_truth.dataset import GroundTruthError
from unproject.run.publish import staged_artifact

EVALUATION_SCHEMA_VERSION = 2

DESCRIPTOR_FILENAME = "evaluation.json"
OBSERVATIONS_FILENAME = "observations.npz"

# Read in chunks so hashing a large sidecar does not load it into memory.
_CHUNK_BYTES = 1024 * 1024


class FileReference(BaseModel):
    """A file this evaluation produced, with enough to know it is intact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Relative to the evaluation directory, so the whole thing can be moved.
    path: str
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetIdentity(BaseModel):
    """Which release was read, per ADR-0004 §4.3."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    release: str
    split: str
    # Over the annotation files, never the images (see `inventory`); the subset
    # manifest has its own digest below.
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)
    subset_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    excluded: list[str] = []
    exclusion_reason: str = ""


class Environment(BaseModel):
    """The machine and the code. Same shape as the benchmark records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    machine: str
    platform: str
    python_version: str
    device: str
    package_versions: dict[str, str]
    git_commit: str | None = None
    git_dirty: bool = False


class Coverage(BaseModel):
    """How much of the dataset actually reached a metric.

    Present on every evaluation because ADR-0004 §6.1 forbids reporting accuracy
    over matched people without it. A model that finds one easy person out of ten
    must not read as accurate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: int = Field(ge=0)
    ground_truth_people: int = Field(ge=0)
    predictions: int = Field(ge=0)
    matched: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    valid_joints: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_the_books_balance(self) -> Self:
        if self.matched + self.false_negatives != self.ground_truth_people:
            msg = (
                f"{self.matched} matched plus {self.false_negatives} missed is not "
                f"{self.ground_truth_people} annotated people; a person was dropped "
                "somewhere, which is exactly what this field exists to catch"
            )
            raise ValueError(msg)
        if self.matched + self.false_positives != self.predictions:
            msg = (
                f"{self.matched} matched plus {self.false_positives} spurious is not "
                f"{self.predictions} predictions"
            )
            raise ValueError(msg)
        return self


class MatchingIdentity(BaseModel):
    """The matcher and the one parameter it read, so a table can quote both.

    The greedy matcher's radius is fixed by its algorithm; the global matcher's
    gate is the config's. Recording the value beside the name is what stops a
    threshold that did nothing from being reported as if it had.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    matcher: MatcherKind
    parameter: Literal["radius_px_per_axis", "gate_fraction_of_image_diagonal"]
    value: float = Field(gt=0.0)


def matching_identity(config: GroundTruthConfig) -> MatchingIdentity:
    if config.matcher == "mupots_greedy_2d_40px":
        return MatchingIdentity(
            matcher=config.matcher,
            parameter="radius_px_per_axis",
            value=MUPOTS_GREEDY_RADIUS_PX,
        )
    gate = config.thresholds.association_gate
    if gate is None:  # pragma: no cover - the config validator forbids this
        raise GroundTruthError("the global matcher has no gate")
    return MatchingIdentity(
        matcher=config.matcher,
        parameter="gate_fraction_of_image_diagonal",
        value=gate,
    )


class BodyArtifactProvenance(BaseModel):
    """Where one sequence's bodies came from.

    The evaluation's own git state says which code scored; this says which code
    inferred, which can be a different commit and a different cleanliness when
    bodies are reused. A final result needs both clean (ADR-0005 WP4).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The per-sequence experiment name, which is the artifact directory.
    experiment: str | None = None
    artifact_hash: str = Field(min_length=1)
    git_commit: str | None = None
    # None when the manifest could not be read, which a v1 descriptor never
    # recorded; distinct from False, which is a positive statement.
    git_dirty: bool | None = None


class Migration(BaseModel):
    """What was changed on the way in, when a v1 file is read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_schema_version: int
    notes: list[str] = []


class EvaluationDescriptor(BaseModel):
    """One evaluation, complete."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=EVALUATION_SCHEMA_VERSION, ge=1)
    evaluation_id: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None

    config: GroundTruthConfig
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    mapping_identifier: str
    mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset: DatasetIdentity
    environment: Environment

    # Repeated from the config so a reader of the file alone sees them without
    # unpacking the config, and so a diff between two descriptors names them.
    metric_protocol: MetricProtocol
    matching: MatchingIdentity
    camera_source: CameraKind

    # Which bodies runs the predictions came from, so a score traces back to the
    # inference that produced it.
    body_artifacts: list[BodyArtifactProvenance] = []
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    # True only when written by a run that refused a dirty tree and dirty body
    # artifacts. Exploratory runs stay False whatever their state happens to be.
    final: bool = False

    coverage: Coverage
    references: list[FileReference] = []
    failures: list[str] = []
    migration: Migration | None = None

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            msg = (
                f"evaluation schema_version {self.schema_version} is not readable by "
                f"this build (expected {EVALUATION_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_the_labels_agree_with_the_config(self) -> Self:
        """The repeated labels are for readers; the config is the record."""
        repeated = (str(self.metric_protocol), str(self.matching.matcher), str(self.camera_source))
        recorded = (
            str(self.config.metric_protocol),
            str(self.config.matcher),
            str(self.config.camera),
        )
        if repeated != recorded:
            msg = f"descriptor labels {repeated} disagree with the config's {recorded}"
            raise ValueError(msg)
        return self

    @property
    def body_artifact_hashes(self) -> list[str]:
        return [artifact.artifact_hash for artifact in self.body_artifacts]


def describe_file(path: Path, *, relative_to: Path) -> FileReference:
    """Size and digest one produced file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return FileReference(
        path=path.relative_to(relative_to).as_posix(),
        byte_size=path.stat().st_size,
        sha256=digest.hexdigest(),
    )


def upgrade_descriptor_data(data: dict[str, object]) -> dict[str, object]:
    """Rewrite a schema-1 descriptor mapping into schema 2. Labels only.

    The config inside is upgraded through `upgrade_config_data`; the identity
    fields the v2 shape repeats are filled from it; the body artifact hashes
    become provenance entries whose git state is unknown, because v1 never
    read the manifests. Every change is listed in `migration.notes`.
    """
    version = data.get("schema_version", 1)
    if version != 1:
        raise GroundTruthError(f"cannot upgrade a schema_version {version} evaluation descriptor")

    upgraded = dict(data)
    raw_config = upgraded.get("config")
    if not isinstance(raw_config, dict):
        raise GroundTruthError("a schema 1 descriptor has no config table to upgrade")
    config_data = upgrade_config_data(dict(raw_config))
    config = GroundTruthConfig.model_validate(config_data)
    notes = [
        f"matcher {raw_config.get('matcher', 'official')!r} read as {config.matcher!r}",
        f"camera {raw_config.get('camera', 'estimated')!r} read as {config.camera!r}",
        f"metric_protocol {config.metric_protocol!r} declared; v1 reported it as official",
    ]
    thresholds = raw_config.get("thresholds")
    if (
        config.matcher == "mupots_greedy_2d_40px"
        and isinstance(thresholds, dict)
        and "association_gate" in thresholds
    ):
        notes.append(
            f"association_gate {thresholds['association_gate']} dropped: the greedy "
            "matcher never read it"
        )

    hashes = upgraded.pop("body_artifact_hashes", [])
    if not isinstance(hashes, list):
        raise GroundTruthError("body_artifact_hashes is not a list")
    upgraded["schema_version"] = EVALUATION_SCHEMA_VERSION
    upgraded["config"] = config_data
    upgraded["metric_protocol"] = config.metric_protocol
    upgraded["matching"] = matching_identity(config).model_dump()
    upgraded["camera_source"] = config.camera
    upgraded["body_artifacts"] = [
        {"artifact_hash": str(value), "git_commit": None, "git_dirty": None} for value in hashes
    ]
    upgraded["final"] = False
    upgraded["migration"] = {"from_schema_version": 1, "notes": notes}
    return upgraded


def read_descriptor(directory: Path) -> EvaluationDescriptor:
    """Load a finished evaluation, checking every file it claims to have."""
    path = directory / DESCRIPTOR_FILENAME
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise GroundTruthError(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise GroundTruthError(f"{path} does not hold a JSON object")
    if data.get("schema_version") == 1:
        data = upgrade_descriptor_data(data)
    try:
        descriptor = EvaluationDescriptor.model_validate(data)
    except ValueError as error:
        raise GroundTruthError(f"cannot read {path}: {error}") from error

    for reference in descriptor.references:
        target = directory / reference.path
        if not target.is_file():
            raise GroundTruthError(
                f"{path} references {reference.path}, which is not there. The "
                "evaluation directory is incomplete; re-run it rather than "
                "reading a partial result."
            )
        actual = target.stat().st_size
        if actual != reference.byte_size:
            raise GroundTruthError(
                f"{reference.path} is {actual} bytes, but {DESCRIPTOR_FILENAME} "
                f"says {reference.byte_size}. It was truncated or replaced."
            )
    return descriptor


@contextmanager
def staged_evaluation(directory: Path) -> Iterator[Path]:
    """Build an evaluation somewhere else, then move it into place.

    Wraps the pipeline's own publication helper so ground-truth results get the
    same guarantee the stages have: a reader never sees half of one.
    """
    with staged_artifact(directory, descriptor=DESCRIPTOR_FILENAME) as staging:
        yield staging


def write_descriptor(descriptor: EvaluationDescriptor, directory: Path) -> Path:
    """Write the completion marker. Call this last, inside the staged block."""
    path = directory / DESCRIPTOR_FILENAME
    path.write_text(descriptor.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
