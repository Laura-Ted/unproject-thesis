"""The evidence bundle: what a reader of the thesis can check without the data.

ADR-0005 WP7. Every number in the thesis comes from an artifact under
`artifacts/`, and most of those artifacts cannot be committed - the footage is
not ours, the dataset's licence forbids redistribution, the weights are 1.6 GB,
the observations may be derived data the licence does not clear. What can be
committed is small: the summaries the tables are read from, the figures'
source data, the descriptors that name every hash, the state of the gates, and
an index of what stayed local with its SHA-256.

A bundle is built from a spec (`experiments/evidence/*.toml`) that lists each
final result by kind and path, with the exact command that produced it. The
builder reads each result, refuses one that is not what the spec says it is -
an evaluation that is not final, a tree that was dirty - copies only the files
its kind allows, none larger than a limit, and writes `release.json`: commit,
gates, and per result its identity, the files carried, and the files indexed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.config import ConfigError
from unproject.gates import GATES_FILENAME, GatesRecord
from unproject.run.manifest import MANIFEST_FILENAME, git_commit, git_dirty

EVIDENCE_SCHEMA_VERSION = 1
RELEASE_FILENAME = "release.json"

# Nothing larger enters a bundle. Figures are tens to hundreds of kilobytes;
# anything past this is data, not evidence.
MAX_FILE_BYTES = 2 * 1024 * 1024

type ResultKind = Literal["ground-truth", "compare", "benchmark-suite", "live", "filter", "rq3-run"]

# What each kind may carry. Globs relative to the result directory.
_ALLOWED: dict[str, tuple[str, ...]] = {
    "ground-truth": (
        "evaluation.json",
        "summary.json",
        "per-sequence.csv",
        "per-joint.csv",
        "figures/*.svg",
        "hypotheses/*.json",
        "hypotheses/*.csv",
        "hypotheses/*.svg",
    ),
    "compare": ("*.txt", "*.csv", "*.svg"),
    "benchmark-suite": ("summary.json", "summary.csv", "pairs.csv", "*suite-run.json", "*/*.json"),
    "live": ("*.json", "summary.csv"),
    "filter": ("*.json", "*.csv", "*.svg"),
    # The run itself is not carried; what is derived from it is (see `_rq3_run`).
    "rq3-run": ("rq3.json", "rq3-positions.csv", "*.svg"),
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ResultSpec(_Frozen):
    kind: ResultKind
    # Also the directory inside the bundle.
    label: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    # Relative to the spec file.
    path: Path
    # The exact command that produced it, for the manifest.
    command: str = Field(min_length=1)
    # A ground-truth evaluation must carry final=true and a clean tree; a
    # benchmark suite, every job from one clean commit; a bodies run, a
    # manifest that records a clean tree. Anything else is not a thesis result.
    require_final: bool = False
    # An absent optional result is recorded as absent rather than failing the
    # build: a bundle can be assembled before every experiment has been run
    # and says which are missing.
    optional: bool = False


class EvidenceSpec(_Frozen):
    schema_version: int = Field(default=EVIDENCE_SCHEMA_VERSION, ge=1)
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    description: str = ""
    results: list[ResultSpec] = Field(min_length=1)
    # Local files that are inputs to everything and must be indexed by hash:
    # weights, footage. Relative to the spec file; an absent one is skipped and
    # leaves no trace in the release.
    inputs: list[Path] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            msg = f"evidence schema_version {self.schema_version} is not supported"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_labels_unique(self) -> Self:
        labels = [result.label for result in self.results]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            msg = f"result labels used more than once: {duplicates}"
            raise ValueError(msg)
        return self


class FileEntry(_Frozen):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class LocalEntry(FileEntry):
    """A file that stays out of the bundle, and why."""

    reason: str


class ResultRecord(_Frozen):
    label: str
    kind: ResultKind
    source: str
    command: str
    status: Literal["collected", "absent"]
    identity: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    files: list[FileEntry] = Field(default_factory=list)
    local_only: list[LocalEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Release(_Frozen):
    schema_version: int = EVIDENCE_SCHEMA_VERSION
    name: str
    description: str
    created_at: datetime
    git_commit: str | None
    git_dirty: bool
    gates_green: bool | None
    results: list[ResultRecord]
    inputs: list[LocalEntry]


class EvidenceError(Exception):
    """Raised when a result is not what the spec says it is."""


def load_evidence_spec(path: Path) -> EvidenceSpec:
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read evidence spec {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error
    try:
        return EvidenceSpec.model_validate(data)
    except ValueError as error:
        raise ConfigError(f"{path} failed validation:\n{error}") from error


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry(path: Path, *, relative_to: Path) -> FileEntry:
    return FileEntry(
        path=path.relative_to(relative_to).as_posix(),
        sha256=sha256_of(path),
        bytes=path.stat().st_size,
    )


def _local(path: Path, *, relative_to: Path, reason: str) -> LocalEntry:
    return LocalEntry(
        path=path.relative_to(relative_to).as_posix()
        if path.is_relative_to(relative_to)
        else str(path),
        sha256=sha256_of(path),
        bytes=path.stat().st_size,
        reason=reason,
    )


def build_bundle(
    spec: EvidenceSpec,
    *,
    spec_dir: Path,
    repo_root: Path,
    out: Path,
    gates: GatesRecord | None = None,
) -> Release:
    """Collect every result into `out` and write the release manifest.

    Built somewhere fresh: an existing bundle directory is replaced whole, so
    a stale file from an earlier build cannot survive beside new ones.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    records: list[ResultRecord] = []
    for result in spec.results:
        source = (spec_dir / result.path).resolve()
        if not source.is_dir():
            if not result.optional:
                raise EvidenceError(
                    f"{result.label}: {source} does not exist. Run\n    {result.command}\n"
                    "or mark the result optional."
                )
            records.append(
                ResultRecord(
                    label=result.label,
                    kind=result.kind,
                    source=result.path.as_posix(),
                    command=result.command,
                    status="absent",
                    notes=["not yet produced"],
                )
            )
            continue
        target = out / result.label
        target.mkdir()
        identity, notes, local_only = _identity(result, source, repo_root)
        files = _copy_allowed(result.kind, source, target, out)
        if result.kind == "rq3-run":
            files.extend(_rq3_run(source, target, out))
        records.append(
            ResultRecord(
                label=result.label,
                kind=result.kind,
                source=result.path.as_posix(),
                command=result.command,
                status="collected",
                identity=identity,
                files=sorted(files, key=lambda entry: entry.path),
                local_only=local_only,
                notes=notes,
            )
        )

    gates_green: bool | None = None
    if gates is not None:
        (out / GATES_FILENAME).write_text(gates.model_dump_json(indent=2) + "\n", encoding="utf-8")
        gates_green = gates.green

    inputs = [
        _local(
            (spec_dir / path).resolve(),
            relative_to=repo_root,
            reason="input; not redistributed here, see its provenance",
        )
        for path in spec.inputs
        if (spec_dir / path).resolve().is_file()
    ]
    release = Release(
        name=spec.name,
        description=spec.description,
        created_at=datetime.now(UTC),
        git_commit=git_commit(),
        git_dirty=git_dirty(),
        gates_green=gates_green,
        results=records,
        inputs=inputs,
    )
    (out / RELEASE_FILENAME).write_text(release.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return release


def _copy_allowed(kind: str, source: Path, target: Path, bundle_root: Path) -> list[FileEntry]:
    entries: list[FileEntry] = []
    for pattern in _ALLOWED[kind]:
        for path in sorted(source.glob(pattern)):
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                raise EvidenceError(
                    f"{path} is {size} bytes, over the {MAX_FILE_BYTES}-byte limit; a bundle "
                    "carries summaries and figures, not data"
                )
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            entries.append(_entry(destination, relative_to=bundle_root))
    return entries


def _identity(
    result: ResultSpec, source: Path, repo_root: Path
) -> tuple[dict[str, str | int | float | bool | None], list[str], list[LocalEntry]]:
    """What identifies the result, what to say about it, and what stays local."""
    if result.kind == "ground-truth":
        return _ground_truth_identity(result, source, repo_root)
    if result.kind == "benchmark-suite":
        return _benchmark_identity(result, source)
    if result.kind == "rq3-run":
        return _rq3_identity(result, source, repo_root)
    return {}, [], []


def _ground_truth_identity(
    result: ResultSpec, source: Path, repo_root: Path
) -> tuple[dict[str, str | int | float | bool | None], list[str], list[LocalEntry]]:
    from unproject.ground_truth.artifacts import OBSERVATIONS_FILENAME, read_descriptor

    descriptor = read_descriptor(source)
    if result.require_final and not descriptor.final:
        raise EvidenceError(
            f"{result.label}: {source.name} is an exploratory evaluation (final=false). A "
            "thesis result is produced with `gt-evaluate --final`; see the WP4 runbook."
        )
    if result.require_final and descriptor.environment.git_dirty:
        raise EvidenceError(f"{result.label}: scored from a dirty tree; not a thesis result")
    dirty_bodies = [a for a in descriptor.body_artifacts if a.git_dirty is not False]
    if result.require_final and dirty_bodies:
        raise EvidenceError(
            f"{result.label}: {len(dirty_bodies)} body artifact(s) from a dirty or unknown tree"
        )
    identity: dict[str, str | int | float | bool | None] = {
        "evaluation_id": descriptor.evaluation_id,
        "config_hash": descriptor.config_hash,
        "metric_protocol": descriptor.metric_protocol,
        "matcher": descriptor.matching.matcher,
        "camera_source": descriptor.camera_source,
        "frame_transform": descriptor.config.frame_transform,
        "final": descriptor.final,
        "git_commit": descriptor.environment.git_commit,
        "git_dirty": descriptor.environment.git_dirty,
        "mapping_identifier": descriptor.mapping_identifier,
        "mapping_sha256": descriptor.mapping_sha256,
        "inventory_sha256": descriptor.dataset.inventory_sha256,
        "checkpoint_sha256": descriptor.checkpoint_sha256,
        "matched": descriptor.coverage.matched,
        "ground_truth_people": descriptor.coverage.ground_truth_people,
        "frames": descriptor.coverage.samples,
    }
    local = [
        _local(
            source / OBSERVATIONS_FILENAME,
            relative_to=repo_root,
            reason="per-person observations derived from a dataset whose licence forbids "
            "redistribution; kept locally, summaries carried",
        )
    ]
    for artifact in descriptor.body_artifacts:
        if artifact.experiment is None:
            continue
        directory = (
            repo_root / "artifacts" / artifact.experiment / "bodies" / artifact.artifact_hash
        )
        for name in ("bodies.json", "bodies.bin", MANIFEST_FILENAME):
            path = directory / name
            if path.is_file():
                local.append(
                    _local(
                        path,
                        relative_to=repo_root,
                        reason="bodies fitted to dataset frames; derived from restricted data",
                    )
                )
    notes = [] if descriptor.final else ["exploratory evaluation, not final"]
    if descriptor.migration is not None:
        notes.append("read from a schema-1 descriptor")
    return identity, notes, local


def _benchmark_identity(
    result: ResultSpec, source: Path
) -> tuple[dict[str, str | int | float | bool | None], list[str], list[LocalEntry]]:
    from unproject.benchmark.summary import read_results

    results, unreadable = read_results(source)
    if not results:
        raise EvidenceError(f"{source} holds no benchmark job results")
    real = [r for r in results if not r.dry_run]
    if result.require_final:
        dirty = sum(1 for r in real if r.environment.git_dirty)
        commits = {r.environment.git_commit for r in real}
        if not real:
            raise EvidenceError(f"{result.label}: only dry runs; not a thesis result")
        if dirty:
            raise EvidenceError(
                f"{result.label}: {dirty} of {len(real)} job(s) ran from a dirty tree; "
                "not a thesis result"
            )
        if len(commits) != 1 or None in commits:
            raise EvidenceError(
                f"{result.label}: jobs from {len(commits)} commit(s) "
                f"({', '.join(sorted(str(c)[:12] for c in commits))}); a suite is one commit"
            )
    identity: dict[str, str | int | float | bool | None] = {
        "suite": real[0].identity.suite if real else results[0].identity.suite,
        "suite_hash": real[0].identity.suite_hash if real else results[0].identity.suite_hash,
        "jobs": len(real),
        "dry_runs": len(results) - len(real),
        "git_commits": ",".join(sorted({r.environment.git_commit or "unknown" for r in real})),
        "git_dirty": any(r.environment.git_dirty for r in real),
        "machines": ",".join(sorted({r.environment.machine for r in real})),
        "checkpoint_sha256": real[0].inputs.checkpoint_sha256 if real else None,
        "source_sha256": ",".join(sorted({r.inputs.source_sha256 for r in real})),
    }
    notes = [f"unreadable: {item}" for item in unreadable]
    if any(r.environment.git_dirty for r in real):
        notes.append("at least one job ran from a dirty tree")
    return identity, notes, []


def _rq3_identity(
    result: ResultSpec, source: Path, repo_root: Path
) -> tuple[dict[str, str | int | float | bool | None], list[str], list[LocalEntry]]:
    from unproject.rq3.reader import read_run

    run = read_run(source)
    artifact = run.artifact
    identity: dict[str, str | int | float | bool | None] = {
        "artifact_hash": source.name,
        "ingest_hash": artifact.ingest_hash,
        "checkpoint_sha256": artifact.estimator.checkpoint_sha256,
        "config_hash": artifact.estimator.config_hash,
        "frames": artifact.summary.frames_total,
        "people": artifact.summary.people_total,
        "tracks": artifact.summary.track_count,
    }
    manifest = source / MANIFEST_FILENAME
    if manifest.is_file():
        data = json.loads(manifest.read_text("utf-8"))
        identity["git_commit"] = data.get("git_commit")
        identity["git_dirty"] = bool(data.get("git_dirty", False))
    if result.require_final and identity.get("git_dirty", True) is not False:
        raise EvidenceError(
            f"{result.label}: {source.name} was inferred from a dirty or unrecorded tree; "
            "not a thesis result"
        )
    local = [
        _local(source / name, relative_to=repo_root, reason="the run itself; derived views carried")
        for name in ("bodies.json", "bodies.bin", MANIFEST_FILENAME)
        if (source / name).is_file()
    ]
    return identity, [], local


def _rq3_run(source: Path, target: Path, bundle_root: Path) -> list[FileEntry]:
    """What the thesis figures need from a bodies run, as small derived files.

    The longest track's pelvis depth, raw and filtered by the analysis's own
    smoother, and the middle frame's skeletons with the camera they project
    through. Both are numbers derived from CC0 footage; neither is the
    footage or the run.
    """
    from unproject.rq3.reader import read_array, read_run
    from unproject.rq3.smoothing import DEFAULT_SMOOTHING, TrackSmoother

    run = read_run(source)
    smoother = TrackSmoother(DEFAULT_SMOOTHING)
    smoothed: dict[tuple[float, int], tuple[float, float, float]] = {}
    for time in sorted({d.time_seconds for d in run.detections}):
        at_time = [d for d in run.detections if d.time_seconds == time]
        for track, root in smoother.frame(
            time,
            [
                (
                    d.track_id,
                    (
                        float(d.root_translation[0]),
                        float(d.root_translation[1]),
                        float(d.root_translation[2]),
                    ),
                )
                for d in at_time
            ],
        ):
            smoothed[(time, track)] = root
    track = max(run.track_ids, key=lambda key: (len(run.track(key)), -key))
    detections = run.track(track)
    depth_csv = target / "track-depth.csv"
    with depth_csv.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["track_id", "frame_index", "time_seconds", "raw_z_metres", "filtered_z_metres"]
        )
        for d in detections:
            writer.writerow(
                [
                    track,
                    d.frame_index,
                    f"{d.time_seconds:.6f}",
                    f"{float(d.root_translation[2]):.6f}",
                    f"{smoothed[(d.time_seconds, track)][2]:.6f}",
                ]
            )

    artifact = run.artifact
    frame = artifact.frames[len(artifact.frames) // 2]
    joints = read_array(source, artifact, "joints_3d")
    frame_json = target / f"frame-{frame.frame_index:06d}.json"
    frame_json.write_text(
        json.dumps(
            {
                "frame_index": frame.frame_index,
                "timestamp_seconds": frame.timestamp_seconds,
                "camera": frame.camera.model_dump(),
                "bone_parents": artifact.body_model.bone_parents,
                "convention": artifact.convention,
                "people": [
                    {
                        "track_id": person.track_id,
                        "joints_3d": np.asarray(joints[person.parameter_index], dtype=np.float64)
                        .round(6)
                        .tolist(),
                    }
                    for person in frame.people
                ],
                "track_selection": "largest number of observations; lowest id breaks ties",
                "selected_track": track,
                "smoothing": DEFAULT_SMOOTHING.model_dump(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [_entry(depth_csv, relative_to=bundle_root), _entry(frame_json, relative_to=bundle_root)]


def verify_bundle(directory: Path) -> list[str]:
    """Every problem with a bundle on disk: a file whose hash moved, or one missing."""
    problems: list[str] = []
    try:
        release = Release.model_validate_json((directory / RELEASE_FILENAME).read_text("utf-8"))
    except (OSError, ValueError) as error:
        return [f"{RELEASE_FILENAME}: {error}"]
    for result in release.results:
        for entry in result.files:
            path = directory / entry.path
            if not path.is_file():
                problems.append(f"{entry.path}: missing")
            elif sha256_of(path) != entry.sha256:
                problems.append(f"{entry.path}: sha256 differs from {RELEASE_FILENAME}")
    return problems


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "MAX_FILE_BYTES",
    "RELEASE_FILENAME",
    "EvidenceError",
    "EvidenceSpec",
    "Release",
    "ResultRecord",
    "ResultSpec",
    "build_bundle",
    "load_evidence_spec",
    "sha256_of",
    "verify_bundle",
]
