"""Running a whole ground-truth evaluation, sequence by sequence.

The order matters and is the same every time: validate the release, then the
mapping, then the estimator settings, and only then load a model. A run that
fails because a joint name is wrong should fail in the first second, not after
an hour of inference.

Nothing here decides what an error means. It pairs people, hands the pairs to
`metrics`, and writes down what came back - so a bug in aggregation cannot be
hidden by a bug in orchestration, and the raw rows survive either.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from unproject.bodies.artifacts import CameraIntrinsics, PersonRecord
from unproject.config import load_config
from unproject.ground_truth.anny_skeleton import anny_bone_labels, check_skeleton
from unproject.ground_truth.artifacts import (
    OBSERVATIONS_FILENAME,
    BodyArtifactProvenance,
    DatasetIdentity,
    Environment,
    EvaluationDescriptor,
    FileReference,
    MatchingIdentity,
    describe_file,
    matching_identity,
    staged_evaluation,
    write_descriptor,
)
from unproject.ground_truth.config import (
    GroundTruthConfig,
    MatcherKind,
    identity_hash,
    resolve,
)
from unproject.ground_truth.dataset import (
    BoolArray,
    DatasetRecord,
    FloatArray,
    GroundTruthError,
)
from unproject.ground_truth.figures import write_figures
from unproject.ground_truth.inventory import inventory
from unproject.ground_truth.mapping import JointMapping, apply_mapping, load_mapping
from unproject.ground_truth.matching import (
    Assignment,
    match_mupots_greedy,
    match_normalized_global,
)
from unproject.ground_truth.metrics import PersonErrors, person_errors
from unproject.ground_truth.mupots import JOINT_NAMES, MupotsDataset
from unproject.ground_truth.observations import (
    Observations,
    collect,
    write_per_joint,
    write_per_sequence,
)
from unproject.ground_truth.report import SUMMARY_FILENAME, build_summary, write_summary
from unproject.ground_truth.sequences import SequenceRun, release_frames, run_sequence
from unproject.ground_truth.transform import letterbox_sequence, release_derived_frames
from unproject.rq3.reader import read_array
from unproject.run.manifest import MANIFEST_FILENAME, RunManifest, git_commit, git_dirty

PER_SEQUENCE_FILENAME = "per-sequence.csv"
PER_JOINT_FILENAME = "per-joint.csv"
FIGURES_DIRECTORY = "figures"

# The joint both root alignment and every absolute metric are defined against.
ROOT_JOINT = "pelvis"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    directory: Path
    descriptor: EvaluationDescriptor
    observations: Observations
    # Bytes of decoded frames released during the run, when asked for.
    frames_released_bytes: int = 0


def evaluate(
    config: GroundTruthConfig,
    *,
    config_dir: Path,
    artifacts_root: Path,
    sequences: list[str] | None = None,
    overwrite: bool = False,
    release_intermediate_frames: bool = False,
    final: bool = False,
) -> EvaluationResult:
    """Score a release end to end and publish the result atomically.

    `final` is the gate ADR-0005 WP4 puts in front of a thesis result: it
    refuses a dirty working tree before loading a model, refuses a subset of
    sequences, and refuses to score bodies whose own manifests record a dirty
    tree or no tree at all. An exploratory run leaves it off and is marked
    so in the descriptor; nothing about it is a thesis number.
    """
    started = datetime.now(UTC)
    if config.dataset != "mupots-3d":
        raise GroundTruthError(f"only mupots-3d is implemented; {config.dataset} is not")
    if final:
        check_final_preconditions(sequences)

    dataset_root = resolve(config.dataset_root, config_dir) / "MultiPersonTestSet"
    found = inventory(config, dataset_root=dataset_root)
    mapping, mapping_sha = load_mapping(resolve(config.mapping, config_dir))
    check_protocol(config, mapping)

    experiment_path = resolve(config.experiment, config_dir)
    base = load_config(experiment_path)
    dataset = MupotsDataset(dataset_root, release_identity=found.content_sha256[:12])

    wanted = sequences if sequences is not None else dataset.sequences()
    unknown = sorted(set(wanted) - set(dataset.sequences()))
    if unknown:
        raise GroundTruthError(f"not on disk: {', '.join(unknown)}")

    # The sequences are part of what this evaluation *is*: a score over one
    # sequence and a score over all twenty are different measurements, and
    # letting them share a directory would let the smaller one be read as the
    # benchmark.
    evaluation_id = hashlib.sha256(
        f"{identity_hash(config, inventory_hash=found.content_sha256, mapping_hash=mapping_sha)}"
        f":{','.join(wanted)}".encode()
    ).hexdigest()
    directory = resolve(config.output_root, config_dir) / evaluation_id[:12]
    if directory.is_dir() and not overwrite:
        raise GroundTruthError(
            f"{directory} already holds a finished evaluation of this exact "
            "configuration and release. Pass --overwrite to replace it."
        )

    people: list[tuple[int, int, int, int, PersonErrors]] = []
    frames: list[tuple[int, int, int, int, int]] = []
    provenance: list[BodyArtifactProvenance] = []
    focal_lengths: list[tuple[float, float]] = []
    checkpoint = ""
    freed = 0
    for index, sequence_id in enumerate(wanted):
        frames_dir = dataset_root / sequence_id
        records = dataset.records(sequence_id)
        intrinsics = dataset.intrinsics(sequence_id)
        padding = 0
        if config.frame_transform == "letterbox_square":
            frames_dir, records, intrinsics, padding = letterbox_sequence(
                records,
                intrinsics,
                artifacts_root=artifacts_root,
                dataset=config.dataset,
                sequence_id=sequence_id,
            )
        run = run_sequence(
            base,
            sequence_id=sequence_id,
            frames_dir=frames_dir,
            records=records,
            intrinsics=intrinsics,
            config_dir=experiment_path.parent,
            artifacts_root=artifacts_root,
            camera=config.camera,
            transform=config.frame_transform,
        )
        if padding:
            print(f"  {sequence_id}: letterboxed to a square, {padding} rows above", flush=True)
        provenance.append(body_provenance(run))
        if final:
            check_final_bodies(provenance[-1])
        checkpoint = run.artifact.estimator.checkpoint_sha256
        # The camera the annotations imply against the one the model invented.
        # Median over frames: the model re-estimates it every frame.
        focal_lengths.append(
            (
                run.intrinsics.camera.fx,
                float(np.median([frame.camera.fx for frame in run.artifact.frames])),
            )
        )
        print(
            f"{'reused' if run.reused else 'fitted'} {sequence_id}: "
            f"{run.artifact.summary.people_total} detection(s) over "
            f"{run.artifact.summary.frames_total} frame(s) against {run.annotated_people} "
            f"annotated; fx annotation-derived {run.intrinsics.camera.fx:.0f}, "
            f"model-estimated {focal_lengths[-1][1]:.0f}",
            flush=True,
        )
        if release_intermediate_frames:
            # Asked for explicitly: the decoded frames are five times the size of
            # the dataset they came from, and nothing after inference reads them.
            freed += release_frames(run, artifacts_root=artifacts_root)
            if padding:
                freed += release_derived_frames(artifacts_root, config.dataset, sequence_id)
        sequence_people, sequence_frames = score_sequence(
            run,
            mapping,
            index,
            matcher=config.matcher,
            gate=config.thresholds.association_gate,
        )
        people.extend(sequence_people)
        frames.extend(sequence_frames)

    observations = collect(mapping.common_names, tuple(wanted), people, frames, focal_lengths)

    matching = matching_identity(config)
    with staged_evaluation(directory) as staging:
        observations.save(staging / OBSERVATIONS_FILENAME)
        references = [describe_file(staging / OBSERVATIONS_FILENAME, relative_to=staging)]
        references.extend(
            write_derived(
                observations,
                config,
                staging,
                evaluation_id=evaluation_id,
                matching=matching,
                final=final,
            )
        )
        descriptor = EvaluationDescriptor(
            evaluation_id=evaluation_id,
            started_at=started,
            finished_at=datetime.now(UTC),
            config=config,
            config_hash=evaluation_id,
            mapping_identifier=mapping.identifier,
            mapping_sha256=mapping_sha,
            dataset=DatasetIdentity(
                name=config.dataset,
                release=config.release,
                split=config.split,
                inventory_sha256=found.content_sha256,
                file_count=found.file_count,
                byte_count=found.byte_count,
                excluded=sorted(set(dataset.sequences()) - set(wanted)),
                exclusion_reason="not requested" if len(wanted) < len(dataset.sequences()) else "",
            ),
            environment=_environment(),
            metric_protocol=config.metric_protocol,
            matching=matching,
            camera_source=config.camera,
            body_artifacts=provenance,
            checkpoint_sha256=checkpoint,
            final=final,
            coverage=observations.coverage,
            references=references,
        )
        write_descriptor(descriptor, staging)

    return EvaluationResult(
        directory=directory,
        descriptor=descriptor,
        observations=observations,
        frames_released_bytes=freed,
    )


def check_final_preconditions(sequences: list[str] | None) -> None:
    """What a final run refuses before it loads anything.

    A dirty tree means the commit does not identify the code that scored; a
    subset means the result is not the benchmark. Both are refused here, in
    the first second, rather than discovered in a descriptor an hour later.
    """
    if git_commit() is None:
        raise GroundTruthError(
            "a final evaluation needs a git commit to record, and this is not a checkout"
        )
    if git_dirty():
        raise GroundTruthError(
            "the working tree is dirty; a final evaluation records a commit that must "
            "identify the code that ran. Commit or stash, then re-run with --final. "
            "Leave --final off for an exploratory run."
        )
    if sequences is not None:
        raise GroundTruthError(
            "--final cannot be combined with --sequence: a subset is not the benchmark"
        )


def check_final_bodies(provenance: BodyArtifactProvenance) -> None:
    """A final score traces to bodies from a clean tree, or it is not final.

    The bodies' manifest is the record of the inference; an unknown state is
    not a clean one.
    """
    where = provenance.experiment or provenance.artifact_hash
    if provenance.git_dirty is None:
        raise GroundTruthError(
            f"{where}: the bodies artifact has no readable manifest, so the state of the "
            "tree that produced it is unknown. Re-run the inference from a clean commit "
            "(pass --overwrite) or leave --final off."
        )
    if provenance.git_dirty:
        raise GroundTruthError(
            f"{where}: the bodies were produced from a dirty tree "
            f"(commit {provenance.git_commit}). A final evaluation cannot score them; "
            "re-run the inference from a clean commit with --overwrite, or leave "
            "--final off."
        )


def write_derived(
    observations: Observations,
    config: GroundTruthConfig,
    directory: Path,
    *,
    evaluation_id: str,
    matching: MatchingIdentity,
    final: bool,
) -> list[FileReference]:
    """Every view of the observations: the tables, the figures, the summary.

    Called at evaluation time and by `gt-summary --regenerate`, and the same
    either way, which is what lets a regenerated table be checked against the
    one the run wrote.
    """
    write_per_sequence(observations, config.thresholds, directory / PER_SEQUENCE_FILENAME)
    write_per_joint(observations, config.thresholds, directory / PER_JOINT_FILENAME)
    figures = write_figures(observations, config.thresholds, directory / FIGURES_DIRECTORY)
    write_summary(
        build_summary(
            observations,
            config.thresholds,
            evaluation_id=evaluation_id,
            metric_protocol=config.metric_protocol,
            matching=matching,
            camera_source=config.camera,
            final=final,
        ),
        directory,
    )
    references = [
        describe_file(directory / name, relative_to=directory)
        for name in (PER_SEQUENCE_FILENAME, PER_JOINT_FILENAME, SUMMARY_FILENAME)
    ]
    references.extend(describe_file(path, relative_to=directory) for path in figures)
    return references


def regenerate_derived(directory: Path) -> tuple[EvaluationDescriptor, list[str]]:
    """Rewrite every derived view of a finished evaluation from its observations.

    No inference, no dataset. The descriptor's references are updated for the
    files rewritten and the observations' reference is left alone, since the
    record is untouched. Returns the new descriptor and the names of the files
    whose bytes changed - empty when the regeneration reproduced the run.
    """
    descriptor, observations = read_evaluation(directory)
    before = {reference.path: reference.sha256 for reference in descriptor.references}
    kept = [
        reference for reference in descriptor.references if reference.path == OBSERVATIONS_FILENAME
    ]
    rewritten = write_derived(
        observations,
        descriptor.config,
        directory,
        evaluation_id=descriptor.evaluation_id,
        matching=descriptor.matching,
        final=descriptor.final,
    )
    updated = descriptor.model_copy(update={"references": [*kept, *rewritten]})
    write_descriptor(updated, directory)
    changed = sorted(
        reference.path for reference in rewritten if before.get(reference.path) != reference.sha256
    )
    return updated, changed


def check_protocol(config: GroundTruthConfig, mapping: JointMapping) -> None:
    """The mapping has to be the one the declared protocol describes.

    A protocol identifier is a promise about joints and root; a mapping with a
    different count under the same identifier would let two incomparable
    results share a name.
    """
    protocol = config.protocol
    if protocol.root_joint not in mapping.common_names:
        raise GroundTruthError(
            f"mapping {mapping.identifier!r} has no {protocol.root_joint!r} joint. Root "
            "alignment and every absolute metric are defined against it."
        )
    if len(mapping.joints) != protocol.common_joints:
        raise GroundTruthError(
            f"metric protocol {protocol.identifier!r} is defined over "
            f"{protocol.common_joints} common joints, but mapping {mapping.identifier!r} "
            f"has {len(mapping.joints)}. A different joint set is a different protocol "
            "and needs its own identifier."
        )


def body_provenance(run: SequenceRun) -> BodyArtifactProvenance:
    """Which code produced one sequence's bodies, read off its manifest.

    Unknown rather than assumed clean when the manifest cannot be read: a
    missing record is a missing record.
    """
    manifest_path = run.artifact_dir / MANIFEST_FILENAME
    commit: str | None = None
    dirty: bool | None = None
    try:
        manifest = RunManifest.model_validate_json(manifest_path.read_text("utf-8"))
    except (OSError, ValueError):
        pass
    else:
        commit, dirty = manifest.git_commit, manifest.git_dirty
    return BodyArtifactProvenance(
        experiment=run.experiment_name,
        artifact_hash=run.artifact_dir.name,
        git_commit=commit,
        git_dirty=dirty,
    )


def score_sequence(
    run: SequenceRun,
    mapping: JointMapping,
    sequence_index: int,
    *,
    matcher: MatcherKind,
    gate: float | None,
) -> tuple[list[tuple[int, int, int, int, PersonErrors]], list[tuple[int, int, int, int, int]]]:
    """Pair and measure one sequence's frames.

    Frames come from the artifact rather than from the annotations, so a frame
    the estimator produced nothing for is still a row saying so.
    """
    check_skeleton(run.artifact.body_model.bone_count, expected=mapping.anny_bone_count)
    predicted = read_array(run.artifact_dir, run.artifact, "joints_3d").astype(np.float64)
    anny_names = anny_bone_labels()
    anny_valid = np.ones(len(anny_names), dtype=np.bool_)
    root_index = mapping.common_names.index(ROOT_JOINT)

    people: list[tuple[int, int, int, int, PersonErrors]] = []
    frames: list[tuple[int, int, int, int, int]] = []
    for frame, record in zip(run.artifact.frames, run.records, strict=True):
        gt_3d, gt_2d, gt_valid = ground_truth_frame(record, mapping)
        pred_3d, pred_valid = _predicted_frame(
            frame.people, predicted, mapping, anny_names, anny_valid
        )
        pred_2d = project_to_image(pred_3d, frame.camera)

        assignment = _match(
            gt_2d,
            gt_valid,
            pred_2d,
            pred_valid,
            record=record,
            matcher=matcher,
            gate=gate,
        )
        scored = 0
        for gt_index, pred_index in assignment.pairs:
            usable = gt_valid[gt_index] & pred_valid[pred_index]
            if not bool(usable[root_index]):
                # Without a root there is nothing to align on and no absolute
                # placement to compare; the person stays a detection, not a score.
                continue
            scored += 1
            people.append(
                (
                    sequence_index,
                    frame.frame_index,
                    gt_index,
                    frame.people[pred_index].track_id,
                    person_errors(
                        gt_3d[gt_index],
                        pred_3d[pred_index],
                        usable,
                        root_index=root_index,
                        names=mapping.common_names,
                    ),
                )
            )
        frames.append(
            (
                sequence_index,
                frame.frame_index,
                len(record.people),
                len(frame.people),
                scored,
            )
        )
    return people, frames


def _match(
    gt_2d: FloatArray,
    gt_valid: BoolArray,
    pred_2d: FloatArray,
    pred_valid: BoolArray,
    *,
    record: DatasetRecord,
    matcher: MatcherKind,
    gate: float | None,
) -> Assignment:
    if matcher == "mupots_greedy_2d_40px":
        return match_mupots_greedy(gt_2d, gt_valid, pred_2d, pred_valid)
    if gate is None:
        raise GroundTruthError(f"matcher {matcher!r} needs an association gate")
    return match_normalized_global(
        gt_2d,
        gt_valid,
        pred_2d,
        pred_valid,
        image_diagonal=float(np.hypot(record.image_width, record.image_height)),
        gate=gate,
    )


def ground_truth_frame(
    record: DatasetRecord, mapping: JointMapping
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Reduce every annotated person in a frame to the common skeleton."""
    joints = len(mapping.joints)
    three = np.zeros((len(record.people), joints, 3))
    two = np.zeros((len(record.people), joints, 2))
    valid = np.zeros((len(record.people), joints), dtype=np.bool_)
    for index, person in enumerate(record.people):
        if person.joints_2d is None:
            raise GroundTruthError(
                f"{record.sample_id} has a person with no 2D joints; association "
                "would have to fall back on the 3D it is meant to be testing"
            )
        three[index], valid[index] = apply_mapping(
            mapping, "mupots", person.joints_3d, JOINT_NAMES, person.joint_valid
        )
        # A 2D midpoint is the midpoint of the two 2D points, so the same
        # reduction applies once the array is padded to the shape it expects.
        padded, _ = apply_mapping(
            mapping,
            "mupots",
            np.column_stack([person.joints_2d, np.zeros(len(JOINT_NAMES))]),
            JOINT_NAMES,
            person.joint_valid,
        )
        two[index] = padded[:, :2]
    return three, two, valid


def _predicted_frame(
    detections: Sequence[PersonRecord],
    joints: FloatArray,
    mapping: JointMapping,
    names: tuple[str, ...],
    valid: BoolArray,
) -> tuple[FloatArray, BoolArray]:
    """Reduce every detection in a frame to the common skeleton."""
    common = len(mapping.joints)
    three = np.zeros((len(detections), common, 3))
    usable = np.zeros((len(detections), common), dtype=np.bool_)
    for index, person in enumerate(detections):
        parameter_index = person.parameter_index
        three[index], usable[index] = apply_mapping(
            mapping, "anny", joints[parameter_index], names, valid
        )
    return three, usable


def project_to_image(joints: FloatArray, camera: CameraIntrinsics) -> FloatArray:
    """Where the estimator's own camera puts its own joints on the image.

    Its 2D and its 3D are consistent by construction, so this is the model's
    actual claim about the image rather than a re-derivation of it.
    """
    depth = joints[..., 2]
    if np.any(depth <= 0.0):
        raise GroundTruthError("a predicted joint is at or behind the camera")
    return np.stack(
        [
            camera.fx * joints[..., 0] / depth + camera.cx,
            camera.fy * joints[..., 1] / depth + camera.cy,
        ],
        axis=-1,
    )


def _environment() -> Environment:
    from unproject.run.manifest import git_commit, git_dirty, package_versions

    return Environment(
        machine=platform.machine(),
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        python_version=platform.python_version(),
        device=_device(),
        package_versions=package_versions(),
        git_commit=git_commit(),
        git_dirty=git_dirty(),
    )


def _device() -> str:
    try:
        import torch
    except ImportError:
        return "unavailable"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def read_evaluation(directory: Path) -> tuple[EvaluationDescriptor, Observations]:
    """Load a finished evaluation. No weights, no dataset, no inference."""
    from unproject.ground_truth.artifacts import read_descriptor

    return read_descriptor(directory), Observations.load(directory / OBSERVATIONS_FILENAME)
