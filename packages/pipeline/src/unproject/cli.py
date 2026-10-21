"""Command line entry point.

Stages are reachable from here and nowhere else, so the command in a lab note
is the command that reproduces the run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from pathlib import Path

from unproject import __version__
from unproject.benchmark.aggregate import read_results, summarise, write_summary
from unproject.benchmark.campaign import (
    DEFAULT_REGISTRY,
    MachineRegistry,
    load_registry,
    read_campaign,
    write_campaign,
)
from unproject.benchmark.config import load_benchmark_config, load_measured_experiment
from unproject.benchmark.dryrun import DryRunEstimator
from unproject.benchmark.job import measure_job
from unproject.benchmark.live import STAGE_NAME as LIVE_STAGE
from unproject.benchmark.live import read_reports
from unproject.benchmark.preflight import print_report, run_preflight, write_report
from unproject.benchmark.run import STAGE_NAME as BENCHMARK_STAGE
from unproject.benchmark.run import (
    BenchmarkError,
    MeasurableEstimator,
    describe_matrix,
    run_benchmark,
)
from unproject.benchmark.suite import (
    Job,
    SuiteConfig,
    load_cell_experiment,
    load_suite_config,
    plan_jobs,
    results_dir,
    run_suite,
)
from unproject.benchmark.summary import read_results as read_suite_results
from unproject.benchmark.summary import summarise as summarise_suite
from unproject.benchmark.summary import write_summary as write_suite_summary
from unproject.bodies.stage import BodiesError, run_bodies
from unproject.config import ConfigError, ExperimentConfig, load_config
from unproject.evaluate.stage import EvaluateError, run_evaluate
from unproject.evidence import (
    RELEASE_FILENAME,
    EvidenceError,
    build_bundle,
    load_evidence_spec,
    verify_bundle,
)
from unproject.gates import GATES_FILENAME, GatesRecord, run_gates, write_gates
from unproject.ground_truth.artifacts import EvaluationDescriptor
from unproject.ground_truth.compare import (
    DIFF_FILENAME,
    compare_evaluations,
    describe,
    write_paired,
)
from unproject.ground_truth.config import (
    PROTOCOLS,
    CameraKind,
    MatcherKind,
    MetricProtocol,
    MetricThresholds,
    load_ground_truth_config,
)
from unproject.ground_truth.dataset import GroundTruthError
from unproject.ground_truth.evaluate import evaluate, read_evaluation, regenerate_derived
from unproject.ground_truth.hypotheses import analyse as analyse_hypotheses
from unproject.ground_truth.hypotheses import write_hypotheses
from unproject.ground_truth.mapping import load_mapping
from unproject.ground_truth.metrics import robust_depth
from unproject.ground_truth.observations import Observations
from unproject.ground_truth.observations import summarise as summarise_ground_truth
from unproject.ingest.source import FrameSourceError
from unproject.ingest.stage import IngestError, run_ingest
from unproject.pose.backend import PoseBackendError
from unproject.pose.stage import PoseError, run_pose
from unproject.preprocess.stage import PreprocessError, run_preprocess
from unproject.rq3.analyse import analyse
from unproject.rq3.analyse import write_summary as write_rq3_summary
from unproject.rq3.annotations import AnnotationError, load_annotation
from unproject.rq3.characterise import characterise, write_characterisation
from unproject.rq3.figures import write_figures
from unproject.rq3.reader import ArtifactReadError, read_run

DEFAULT_ARTIFACTS_ROOT = Path("artifacts")

logger = logging.getLogger("unproject")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=bool(args.verbose))

    try:
        if args.command == "validate-config":
            return _cmd_validate_config(args.paths)
        if args.command == "ingest":
            return _cmd_ingest(args.config, args.artifacts_root, overwrite=bool(args.overwrite))
        if args.command == "preprocess":
            return _cmd_preprocess(args.config, args.artifacts_root, overwrite=bool(args.overwrite))
        if args.command == "pose":
            return _cmd_pose(args.config, args.artifacts_root, overwrite=bool(args.overwrite))
        if args.command == "bodies":
            return _cmd_bodies(args.config, args.artifacts_root, overwrite=bool(args.overwrite))
        if args.command == "evaluate":
            return _cmd_evaluate(args.reference, args.candidate, args.artifacts_root)
        if args.command == "benchmark":
            return _cmd_benchmark(args.config, args.artifacts_root, dry_run=bool(args.dry_run))
        if args.command == "benchmark-summary":
            return _cmd_benchmark_summary(args.results)
        if args.command == "benchmark-suite":
            return _cmd_benchmark_suite(
                args.suite,
                args.artifacts_root,
                dry_run=bool(args.dry_run),
                only=args.cell,
                machine_id=args.machine,
            )
        if args.command == "benchmark-preflight":
            return _cmd_benchmark_preflight(
                args.suite,
                args.artifacts_root,
                machine_id=args.machine,
                registry_path=args.registry,
                fetch_weights=bool(args.fetch_weights),
                strict=bool(args.strict),
            )
        if args.command == "benchmark-campaign-summary":
            return _cmd_benchmark_campaign_summary(args.root, registry_path=args.registry)
        if args.command == "benchmark-job":
            return _cmd_benchmark_job(
                args.suite,
                args.artifacts_root,
                cell=args.cell,
                repetition=args.repetition,
                job_index=args.job_index,
                job_count=args.job_count,
                dry_run=bool(args.dry_run),
            )
        if args.command == "benchmark-suite-summary":
            return _cmd_benchmark_suite_summary(args.suite, args.artifacts_root)
        if args.command == "live-summary":
            return _cmd_live_summary(args.reports)
        if args.command == "gt-evaluate":
            return _cmd_gt_evaluate(
                args.config,
                args.artifacts_root,
                sequences=args.sequence,
                matcher=args.matcher,
                overwrite=bool(args.overwrite),
                release_frames=bool(args.release_frames),
                final=bool(args.final),
            )
        if args.command == "gt-summary":
            return _cmd_gt_summary(args.evaluation, regenerate=bool(args.regenerate))
        if args.command == "gt-compare":
            return _cmd_gt_compare(
                args.a, args.b, out=args.out, label_a=args.label_a, label_b=args.label_b
            )
        if args.command == "gt-hypotheses":
            return _cmd_gt_hypotheses(args.evaluation, out=args.out)
        if args.command == "rq3":
            return _cmd_rq3(args.run, args.annotations, args.out)
        if args.command == "filter-characterise":
            return _cmd_filter_characterise(args.out)
        if args.command == "gates":
            return _cmd_gates(args.root, args.record)
        if args.command == "evidence-bundle":
            return _cmd_evidence_bundle(
                args.spec,
                args.out,
                root=args.root,
                gates=args.gates,
                run_gates_first=bool(args.run_gates),
            )
        if args.command == "evidence-verify":
            return _cmd_evidence_verify(args.bundle)
    except (
        AnnotationError,
        ArtifactReadError,
        BenchmarkError,
        ConfigError,
        IngestError,
        PreprocessError,
        PoseError,
        BodiesError,
        EvaluateError,
        EvidenceError,
        FrameSourceError,
        PoseBackendError,
        GroundTruthError,
    ) as error:
        logger.error("%s", error)
        return 1

    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unproject", description=__doc__)
    parser.add_argument("--version", action="version", version=f"unproject {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log at DEBUG level")
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser(
        "validate-config",
        help="parse and validate experiment configs; touches no data",
    )
    validate.add_argument("paths", nargs="+", type=Path, help="config files or directories")

    ingest = subparsers.add_parser(
        "ingest",
        help="stage [1]: decode the configured source into frames plus metadata",
    )
    ingest.add_argument("config", type=Path, help="experiment config (TOML)")
    _add_artifacts_root(ingest)
    ingest.add_argument(
        "--overwrite",
        action="store_true",
        help="re-decode even if a complete artifact already exists",
    )

    preprocess = subparsers.add_parser(
        "preprocess",
        help="stage [2]: reduce the ingested frames to keyframes; runs stage [1] first",
    )
    preprocess.add_argument("config", type=Path, help="experiment config (TOML)")
    _add_artifacts_root(preprocess)
    preprocess.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run even if a complete artifact already exists",
    )

    pose = subparsers.add_parser(
        "pose",
        help="stage [3]: estimate camera poses; runs stages [1] and [2] first",
    )
    pose.add_argument("config", type=Path, help="experiment config (TOML)")
    _add_artifacts_root(pose)
    pose.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run even if a complete artifact already exists",
    )

    bodies = subparsers.add_parser(
        "bodies",
        help="stage [B]: fit body models to the people in every frame; runs stage [1] first",
    )
    bodies.add_argument("config", type=Path, help="experiment config (TOML)")
    _add_artifacts_root(bodies)
    bodies.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run even if a complete artifact already exists",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="stage [5]: measure one pose backend against the other; both must have run",
    )
    evaluate.add_argument("reference", type=Path, help="baseline experiment config (TOML)")
    evaluate.add_argument("candidate", type=Path, help="config to measure against it (TOML)")
    _add_artifacts_root(evaluate)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="measure one cell of the RQ1/RQ2 matrix and write a raw result",
    )
    benchmark.add_argument("config", type=Path, help="benchmark config (TOML)")
    _add_artifacts_root(benchmark)
    benchmark.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise the orchestration with a stand-in estimator; measures nothing real",
    )

    summary = subparsers.add_parser(
        "benchmark-summary",
        help="regenerate summary.json and summary.csv from raw results; runs no inference",
    )
    summary.add_argument(
        "results",
        type=Path,
        nargs="?",
        default=DEFAULT_ARTIFACTS_ROOT / BENCHMARK_STAGE,
        help=f"directory of raw results (default: {DEFAULT_ARTIFACTS_ROOT / BENCHMARK_STAGE})",
    )

    suite = subparsers.add_parser(
        "benchmark-suite",
        help="protocol v2: measure every cell of a suite, each repetition in a fresh process",
    )
    suite.add_argument("suite", type=Path, help="benchmark suite config (TOML)")
    _add_artifacts_root(suite)
    suite.add_argument(
        "--cell",
        action="append",
        default=None,
        metavar="NAME",
        help="run only this cell's jobs, in their planned order; repeatable",
    )
    suite.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise the orchestration with a stand-in estimator; measures nothing real",
    )
    suite.add_argument(
        "--machine",
        metavar="ID",
        default=None,
        help="the declared machine this runs on (benchmarks/machines.toml); written into every job",
    )

    preflight = subparsers.add_parser(
        "benchmark-preflight",
        help="ADR-0006: check a machine is ready to run a suite; read-only, writes one report",
    )
    preflight.add_argument("suite", type=Path, help="benchmark suite config (TOML)")
    preflight.add_argument("--machine", metavar="ID", required=True, help="declared machine id")
    _add_artifacts_root(preflight, "where the suite will write")
    preflight.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"machine registry (default: {DEFAULT_REGISTRY})",
    )
    preflight.add_argument(
        "--fetch-weights",
        action="store_true",
        help="fetch the pinned checkpoint now if absent, rather than in the first job",
    )
    preflight.add_argument(
        "--strict", action="store_true", help="exit non-zero on warnings as well as failures"
    )

    campaign = subparsers.add_parser(
        "benchmark-campaign-summary",
        help="ADR-0006: fold every machine's suite summaries under a root into one table",
    )
    campaign.add_argument("root", type=Path, help="campaign root holding <machine>/benchmark-v2/")
    campaign.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"machine registry (default: {DEFAULT_REGISTRY}); missing is tolerated",
    )

    job = subparsers.add_parser(
        "benchmark-job",
        help="protocol v2, one job: run by benchmark-suite in a process of its own",
    )
    job.add_argument("suite", type=Path, help="benchmark suite config (TOML)")
    job.add_argument("--cell", required=True, help="cell name")
    job.add_argument("--repetition", type=int, required=True)
    job.add_argument("--job-index", type=int, required=True, help="position in the run order")
    job.add_argument("--job-count", type=int, required=True)
    _add_artifacts_root(job)
    job.add_argument("--dry-run", action="store_true", help="stand-in estimator")

    suite_summary = subparsers.add_parser(
        "benchmark-suite-summary",
        help="protocol v2: regenerate summary.json, summary.csv and pairs.csv; no inference",
    )
    suite_summary.add_argument("suite", type=Path, help="benchmark suite config (TOML)")
    _add_artifacts_root(suite_summary, "where the results were written")

    live_summary = subparsers.add_parser(
        "live-summary",
        help="tabulate the live end-to-end reports under artifacts/live; no inference",
    )
    live_summary.add_argument(
        "reports",
        type=Path,
        nargs="?",
        default=DEFAULT_ARTIFACTS_ROOT / LIVE_STAGE,
        help=f"directory of live reports (default: {DEFAULT_ARTIFACTS_ROOT / LIVE_STAGE})",
    )

    gt_evaluate = subparsers.add_parser(
        "gt-evaluate",
        help="score a public ground-truth release; runs stage [B] per sequence, reusing it",
    )
    gt_evaluate.add_argument("config", type=Path, help="ground-truth evaluation config (TOML)")
    gt_evaluate.add_argument(
        "--sequence",
        action="append",
        default=None,
        metavar="ID",
        help="evaluate only this sequence; repeatable. Default: every sequence on disk",
    )
    gt_evaluate.add_argument(
        "--matcher",
        choices=["mupots_greedy_2d_40px", "normalized_global_assignment"],
        default=None,
        help="person association: the released evaluator's greedy 40 px per axis, or "
        "minimum-cost assignment on normalised distance, which needs "
        "thresholds.association_gate in the config (default: the config's own)",
    )
    _add_artifacts_root(gt_evaluate, "where body artifacts are written")
    gt_evaluate.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a finished evaluation of this exact config and release",
    )
    gt_evaluate.add_argument(
        "--release-frames",
        action="store_true",
        help="delete each sequence's decoded frames once its bodies are written. "
        "Nothing measured is lost - the frames are lossless copies of dataset "
        "files that stay on disk - and peak usage drops from the whole release "
        "to one sequence",
    )

    gt_evaluate.add_argument(
        "--final",
        action="store_true",
        help="a thesis result: refuse a dirty tree, a subset of sequences, and bodies "
        "whose manifests record a dirty tree; mark the descriptor final",
    )

    gt_summary = subparsers.add_parser(
        "gt-summary",
        help="report a finished ground-truth evaluation; no inference, no weights, no dataset",
    )
    gt_summary.add_argument("evaluation", type=Path, help="a finished evaluation directory")
    gt_summary.add_argument(
        "--regenerate",
        action="store_true",
        help="rewrite the tables, figures and summary.json from observations.npz and say "
        "which, if any, changed",
    )

    gt_compare = subparsers.add_parser(
        "gt-compare",
        help="two evaluations side by side: same numbers under new labels, or the paired "
        "per-sequence effect of an intervention; no inference",
    )
    gt_compare.add_argument("a", type=Path, help="a finished evaluation directory")
    gt_compare.add_argument("b", type=Path, help="another")
    gt_compare.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the diff, the paired table and figure "
        "(default: artifacts/ground-truth/compare-<a>-<b>)",
    )
    gt_compare.add_argument("--label-a", default="a", help="column and legend label for a")
    gt_compare.add_argument("--label-b", default="b", help="column and legend label for b")

    gt_hypotheses = subparsers.add_parser(
        "gt-hypotheses",
        help="test the body-scale hypothesis on a finished evaluation, pre-registered "
        "reading rule; post-hoc diagnostics labelled; no inference",
    )
    gt_hypotheses.add_argument("evaluation", type=Path, help="a finished evaluation directory")
    gt_hypotheses.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write hypotheses.json, the CSV and the figure "
        "(default: <evaluation>/hypotheses)",
    )

    rq3 = subparsers.add_parser(
        "rq3",
        help="score a bodies run against a controlled scene's annotations; runs no inference",
    )
    rq3.add_argument("run", type=Path, help="a finished bodies artifact directory")
    rq3.add_argument("annotations", type=Path, help="annotation file for that recording (TOML)")
    rq3.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the summary (default: beside the run)",
    )

    characterise_parser = subparsers.add_parser(
        "filter-characterise",
        help="run the viewer's root filter over synthetic signals with a known truth; "
        "no inference, deterministic",
    )
    characterise_parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT / "filter",
        help=f"where to write the report (default: {DEFAULT_ARTIFACTS_ROOT / 'filter'})",
    )

    gates = subparsers.add_parser(
        "gates",
        help="run every quality gate the CI runs, from the repository root, and record "
        "how each ended",
    )
    gates.add_argument("--root", type=Path, default=Path(), help="repository root")
    gates.add_argument(
        "--record",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT / GATES_FILENAME,
        help=f"where to write the record (default: {DEFAULT_ARTIFACTS_ROOT / GATES_FILENAME})",
    )

    bundle = subparsers.add_parser(
        "evidence-bundle",
        help="collect the final results named by a spec into a small, versionable bundle",
    )
    bundle.add_argument("spec", type=Path, help="evidence spec (TOML)")
    bundle.add_argument("out", type=Path, help="bundle directory; replaced whole")
    bundle.add_argument("--root", type=Path, default=Path(), help="repository root")
    bundle.add_argument(
        "--gates",
        type=Path,
        default=None,
        help="a gates.json to carry; see `unproject gates`",
    )
    bundle.add_argument(
        "--run-gates",
        action="store_true",
        help="run the gates now and carry the record (several minutes)",
    )

    verify = subparsers.add_parser(
        "evidence-verify",
        help="check every file a bundle carries against the hashes in its release.json",
    )
    verify.add_argument("bundle", type=Path, help="bundle directory")
    return parser


def _add_artifacts_root(
    parser: argparse.ArgumentParser, what: str = "where artifacts are written"
) -> None:
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help=f"{what} (default: {DEFAULT_ARTIFACTS_ROOT})",
    )


def _cmd_validate_config(paths: list[Path]) -> int:
    configs = _collect_configs(paths)
    if not configs:
        logger.error("no .toml config found in %s", ", ".join(str(p) for p in paths))
        return 1

    failures = 0
    for path in configs:
        try:
            print(f"ok    {path}  ({_describe_config(path)})")
        except (AnnotationError, ConfigError, GroundTruthError) as error:
            # Report every broken config in one pass rather than one per CI run.
            failures += 1
            logger.error("%s", error)

    if failures:
        logger.error("%d of %d config(s) failed validation", failures, len(configs))
    return 1 if failures else 0


def _describe_config(path: Path) -> str:
    """Validate one config against whichever schema it is, and name it.

    `experiments/` holds seven kinds now, told apart by content rather than by
    location: an evidence spec lists results, a joint mapping declares the body
    model it was written against, a ground-truth evaluation names a dataset and
    its root, a benchmark suite lists cells, a v1 benchmark names the
    experiment it measures, an annotation declares a coordinate convention, and
    an experiment has an `[ingest]` section.
    """
    try:
        data = tomllib.loads(path.read_bytes().decode("utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error

    # An evidence spec is the only kind that lists results.
    if "results" in data:
        spec = load_evidence_spec(path)
        optional = sum(1 for result in spec.results if result.optional)
        return (
            f"evidence={spec.name}, {len(spec.results)} result(s), {optional} optional, "
            f"{len(spec.inputs)} input(s) indexed"
        )
    # A suite is the only kind with cells.
    if "cells" in data:
        suite = load_suite_config(path)
        return (
            f"benchmark-suite={suite.name}, {len(suite.cells)} cell(s) x "
            f"{suite.repetitions} repetition(s), {suite.measured_frames} measured frame(s)"
        )
    # A joint mapping declares the body model it was written against, which no
    # other kind of file here does.
    if "anny_bone_count" in data:
        mapping, _ = load_mapping(path)
        return f"mapping={mapping.identifier}, {len(mapping.joints)} common joint(s)"
    # Before the benchmark branch: a ground-truth config also names an
    # `experiment`, so testing for that first would misread it as a benchmark.
    if "dataset" in data and "dataset_root" in data:
        ground_truth = load_ground_truth_config(path)
        return (
            f"ground-truth={ground_truth.name}, dataset={ground_truth.dataset}, "
            f"split={ground_truth.split}, protocol={ground_truth.metric_protocol}, "
            f"matcher={ground_truth.matcher}, camera={ground_truth.camera}"
        )
    # `convention` rather than `markers`: a bare `markers = []` written after a
    # table header belongs to that table, and an annotation with no markers in
    # it would then be read as an experiment and fail confusingly.
    if "convention" in data or "markers" in data:
        annotation = load_annotation(path)
        return (
            f"annotations={annotation.name}, {len(annotation.markers)} marker(s), "
            f"{len(annotation.stationary)} stationary, {len(annotation.motion)} walk(s)"
        )
    if "experiment" in data:
        benchmark = load_benchmark_config(path)
        return f"benchmark={benchmark.name}, output={benchmark.output}"
    return f"experiment={load_config(path).name}"


def _cmd_ingest(config_path: Path, artifacts_root: Path, *, overwrite: bool) -> int:
    config = load_config(config_path)
    result = run_ingest(
        config,
        config_dir=config_path.resolve().parent,
        artifacts_root=artifacts_root,
        overwrite=overwrite,
    )

    selection = result.artifact.selection
    status = "reused" if result.reused else "wrote"
    print(f"{status} {selection.frames_written} frame(s) in {result.artifact_dir}")
    if selection.truncated:
        print(f"note: stopped at max_frames={selection.max_frames}; the source has more")
    return 0


def _cmd_preprocess(config_path: Path, artifacts_root: Path, *, overwrite: bool) -> int:
    config = load_config(config_path)
    result = run_preprocess(
        config,
        config_dir=config_path.resolve().parent,
        artifacts_root=artifacts_root,
        overwrite=overwrite,
    )

    selection = result.artifact.selection
    status = "reused" if result.reused else "kept"
    print(
        f"{status} {selection.keyframes_kept} of {selection.frames_considered} frame(s) "
        f"in {result.artifact_dir}"
    )
    if selection.held_out_every is None:
        print("note: no held-out split; stage [5] cannot score this run honestly")
    else:
        training = selection.keyframes_kept - selection.held_out_count
        print(
            f"      {training} for training, {selection.held_out_count} held out "
            f"(every {selection.held_out_every})"
        )
    if selection.forced_keyframes:
        print(
            f"note: {selection.forced_keyframes} keyframe(s) kept by the blur safety net; "
            "the footage has a stretch with nothing sharp in it"
        )
    return 0


def _cmd_pose(config_path: Path, artifacts_root: Path, *, overwrite: bool) -> int:
    config = load_config(config_path)
    result = run_pose(
        config,
        config_dir=config_path.resolve().parent,
        artifacts_root=artifacts_root,
        overwrite=overwrite,
    )

    summary = result.artifact.registration
    status = "reused" if result.reused else "registered"
    print(
        f"{status} {summary.keyframes_registered} of {summary.keyframes_total} keyframe(s) "
        f"with {result.artifact.backend.name} in {result.artifact_dir}"
    )
    if summary.keyframes_registered < summary.keyframes_total:
        missing = summary.keyframes_total - summary.keyframes_registered
        print(f"note: {missing} keyframe(s) could not be placed and carry no pose")
    if summary.mean_reprojection_error_px is not None:
        print(f"      mean reprojection error {summary.mean_reprojection_error_px:.3f} px")
    print(f"      {summary.points_count} sparse points")
    return 0


def _cmd_bodies(config_path: Path, artifacts_root: Path, *, overwrite: bool) -> int:
    config = load_config(config_path)
    result = run_bodies(
        config,
        config_dir=config_path.resolve().parent,
        artifacts_root=artifacts_root,
        overwrite=overwrite,
    )

    summary = result.artifact.summary
    status = "reused" if result.reused else "fitted"
    print(
        f"{status} {summary.people_total} body/bodies over {summary.frames_total} frame(s) "
        f"in {result.artifact_dir}"
    )
    print(
        f"      up to {summary.max_people_in_frame} in one frame, "
        f"{summary.track_count} distinct identities"
    )
    if summary.frames_with_people < summary.frames_total:
        empty = summary.frames_total - summary.frames_with_people
        print(f"note: {empty} frame(s) held nobody above confidence {summary.confidence_threshold}")
    # Cheapest signal that tracking has stopped making sense.
    if summary.track_count > 4 * max(summary.max_people_in_frame, 1):
        print(
            f"note: {summary.track_count} identities for at most "
            f"{summary.max_people_in_frame} people in frame; tracking is churning"
        )
    return 0


def _cmd_evaluate(reference_path: Path, candidate_path: Path, artifacts_root: Path) -> int:
    result = run_evaluate(
        load_config(reference_path),
        load_config(candidate_path),
        reference_dir=reference_path.resolve().parent,
        candidate_dir=candidate_path.resolve().parent,
        artifacts_root=artifacts_root,
    )

    pose = result.artifact.pose
    errors = pose.errors
    print(
        f"{pose.candidate.backend} against {pose.reference.backend} "
        f"on {pose.keyframes_compared} keyframe(s) in {result.artifact_dir}"
    )
    if pose.keyframes_reference_only or pose.keyframes_candidate_only:
        print(
            f"note: {pose.keyframes_reference_only} placed only by "
            f"{pose.reference.backend}, {pose.keyframes_candidate_only} only by "
            f"{pose.candidate.backend}"
        )
    print(
        f"      position  mean {errors.translation_mean:.4f}  "
        f"median {errors.translation_median:.4f}  max {errors.translation_max:.4f}"
    )
    print(
        f"                {100.0 * errors.translation_mean_fraction:.2f}% of the "
        f"{pose.reference_extent:.3f} unit trajectory"
    )
    print(
        f"      rotation  mean {errors.rotation_mean_degrees:.3f} deg  "
        f"median {errors.rotation_median_degrees:.3f}  max {errors.rotation_max_degrees:.3f}"
    )
    # Free parameter, not an error: neither backend knows the real scale.
    print(f"      scale     {pose.scale:.4f}")
    return 0


def _cmd_benchmark(config_path: Path, artifacts_root: Path, *, dry_run: bool) -> int:
    config = load_benchmark_config(config_path)
    config_dir = config_path.resolve().parent
    experiment, experiment_dir = load_measured_experiment(config, config_dir)

    estimator: MeasurableEstimator | None = None
    if dry_run:
        # Deliberately loud: a dry run leaves a result file that looks like
        # every other one; only its `dry_run` flag and all-zero checkpoint
        # digest say the model never ran.
        print("dry run: the estimator is a stand-in, so these timings measure nothing")
        estimator = DryRunEstimator(experiment.bodies)

    result, path = run_benchmark(
        config,
        config_dir=config_dir,
        artifacts_root=artifacts_root,
        estimator=estimator,
        dry_run=dry_run,
    )
    print(describe_matrix(config, experiment))
    print(f"wrote {len(result.frames)} sample(s) to {path}")
    if result.status == "failed":
        print(f"failed: {result.error}")
        return 1

    steady = result.steady_state()
    if steady:
        ordered = sorted(sample.seconds for sample in steady)
        middle = ordered[len(ordered) // 2]
        print(
            f"      {len(ordered)} steady-state frame(s), median {middle:.4f} s "
            f"({1.0 / middle:.2f} fps) on {result.environment.machine} via "
            f"{result.environment.device}"
        )
    cold = [rep.cold_start_seconds for rep in result.repetitions if rep.index == 0]
    if cold:
        print(f"      cold start {cold[0]:.2f} s, reported apart from the frames above")
    if experiment_dir != config_dir:
        logger.debug("measured %s from %s", experiment.name, experiment_dir)
    return 0


def _cmd_benchmark_summary(results_root: Path) -> int:
    if not results_root.is_dir():
        logger.error("no benchmark results at %s; run `unproject benchmark` first", results_root)
        return 1

    results, unreadable = read_results(results_root)
    if not results and not unreadable:
        logger.error("no benchmark results under %s", results_root)
        return 1

    summary = summarise(results, unreadable)
    as_json, as_csv = write_summary(summary, results_root)
    print(f"summarised {len(results)} result file(s) into {as_json} and {as_csv}")
    if summary.dry_runs_excluded:
        print(f"      excluded {summary.dry_runs_excluded} dry run(s), which measure nothing")
    for benchmark in summary.benchmarks:
        latency = benchmark.latency
        median = "n/a" if latency.median_seconds is None else f"{latency.median_seconds:.4f} s"
        tail = "n/a" if latency.p95_seconds is None else f"{latency.p95_seconds:.4f} s"
        print(
            f"      {benchmark.name} ({benchmark.output}): {latency.samples} sample(s), "
            f"median {median}, p95 {tail}"
        )
        # Exclusions stay visible.
        for problem in benchmark.errors:
            print(f"      note: {benchmark.name} recorded a failure: {problem}")
    return 0


def _cmd_benchmark_suite(
    suite_path: Path,
    artifacts_root: Path,
    *,
    dry_run: bool,
    only: list[str] | None,
    machine_id: str | None = None,
) -> int:
    config = load_suite_config(suite_path)
    if dry_run:
        print("dry run: every job uses a stand-in estimator, so these timings measure nothing")
    jobs = plan_jobs(config)
    where = f" on machine {machine_id}" if machine_id else ""
    print(
        f"{config.name}: {len(config.cells)} cell(s) x {config.repetitions} repetition(s) = "
        f"{len(jobs)} job(s), order seed {config.order_seed}, device {config.device}, "
        f"each in its own process{where}"
    )
    record = run_suite(
        config,
        suite_path=suite_path.resolve(),
        artifacts_root=artifacts_root,
        dry_run=dry_run,
        only=only,
        machine_id=machine_id,
    )
    failed = [outcome for outcome in record.jobs if outcome.exit_code != 0]
    print(f"ran {len(record.jobs)} job(s) into {results_dir(artifacts_root, config.name)}")
    for outcome in failed:
        # Exclusions stay visible, as the protocol requires.
        print(
            f"note: job {outcome.index} ({outcome.cell}, repetition {outcome.repetition}) "
            f"exited {outcome.exit_code}"
        )
    return 1 if failed else 0


def _cmd_benchmark_job(
    suite_path: Path,
    artifacts_root: Path,
    *,
    cell: str,
    repetition: int,
    job_index: int,
    job_count: int,
    dry_run: bool,
) -> int:
    config = load_suite_config(suite_path)
    estimator: MeasurableEstimator | None = None
    if dry_run:
        experiment, _ = _cell_experiment(config, cell, suite_path.resolve().parent)
        estimator = DryRunEstimator(experiment.bodies)
    result, path = measure_job(
        config,
        Job(index=job_index, cell=cell, repetition=repetition),
        suite_dir=suite_path.resolve().parent,
        artifacts_root=artifacts_root,
        job_count=job_count,
        estimator=estimator,
        dry_run=dry_run,
    )
    steady = result.steady_state()
    ordered = sorted(sample.inference_seconds for sample in steady)
    median = ordered[len(ordered) // 2] if ordered else float("nan")
    print(
        f"{cell} repetition {repetition}: {len(steady)} steady frame(s), inference median "
        f"{median:.4f} s, model load {result.phases.model_load_seconds:.2f} s, "
        f"{result.geometry.model_input_width}x{result.geometry.model_input_height} into the "
        f"model, on {result.environment.machine} via {result.environment.device}"
    )
    print(f"wrote {path}")
    if result.status == "failed":
        print(f"failed: {result.error}")
        return 1
    return 0


def _cell_experiment(
    config: SuiteConfig, cell: str, suite_dir: Path
) -> tuple[ExperimentConfig, Path]:
    return load_cell_experiment(config, config.cell(cell), suite_dir)


def _cmd_benchmark_suite_summary(suite_path: Path, artifacts_root: Path) -> int:
    config = load_suite_config(suite_path)
    directory = results_dir(artifacts_root, config.name)
    if not directory.is_dir():
        logger.error("no results at %s; run `unproject benchmark-suite` first", directory)
        return 1
    results, unreadable = read_suite_results(directory)
    if not results and not unreadable:
        logger.error("no job results under %s", directory)
        return 1
    summary = summarise_suite(config.name, results, pairs=config.pairs, unreadable=unreadable)
    as_json, as_csv, as_pairs = write_suite_summary(summary, directory)
    print(f"summarised {len(results)} job(s) into {as_json}, {as_csv} and {as_pairs}")
    if summary.dry_runs_excluded:
        print(f"      excluded {summary.dry_runs_excluded} dry run(s), which measure nothing")
    for problem in summary.unreadable:
        print(f"      note: unreadable result {problem}")
    for cell in summary.cells:
        median = "n/a" if cell.inference.median is None else f"{cell.inference.median:.4f} s"
        tail = "n/a" if cell.inference.p95 is None else f"{cell.inference.p95:.4f} s"
        fps = (
            "n/a" if cell.throughput_fps_median is None else f"{cell.throughput_fps_median:.2f} fps"
        )
        short = "" if cell.meets_protocol else "  [below protocol minimums]"
        print(
            f"      {cell.name} ({cell.output}): {cell.inference.samples} sample(s) over "
            f"{cell.jobs} job(s), inference median {median}, p95 {tail}, {fps}{short}"
        )
        for problem in cell.errors:
            print(f"      note: {cell.name} recorded a failure: {problem}")
    for pair in summary.pairs:
        delta = "n/a" if pair.difference.median is None else f"{pair.difference.median:+.4f} s"
        frames = "" if pair.same_frames else "  [frames differ]"
        print(
            f"      {pair.b} - {pair.a} ({pair.question}): median {delta} over "
            f"{pair.pairs} paired frame(s){frames}"
        )
    return 0


def _cmd_benchmark_preflight(
    suite_path: Path,
    artifacts_root: Path,
    *,
    machine_id: str,
    registry_path: Path,
    fetch_weights: bool,
    strict: bool,
) -> int:
    config = load_suite_config(suite_path)
    registry: MachineRegistry | None = None
    if registry_path.is_file():
        registry = load_registry(registry_path)
    else:
        logger.warning("no machine registry at %s; the machine id is not checked", registry_path)
    report = run_preflight(
        config,
        suite_dir=suite_path.resolve().parent,
        artifacts_root=artifacts_root,
        machine_id=machine_id,
        registry=registry,
        fetch_weights=fetch_weights,
    )
    print_report(report)
    path = write_report(report, artifacts_root)
    print(f"wrote {path}")
    if not report.ok:
        return 1
    if strict and any(check.status == "warn" for check in report.checks):
        return 1
    return 0


def _cmd_benchmark_campaign_summary(root: Path, *, registry_path: Path) -> int:
    registry = load_registry(registry_path) if registry_path.is_file() else None
    summary = read_campaign(root, registry=registry)
    as_json, as_csv, as_pairs = write_campaign(summary, root)
    print(
        f"{len(summary.cells)} cell(s) from {len(summary.machines)} machine(s) and "
        f"{len(summary.suites)} suite(s) into {as_json}, {as_csv} and {as_pairs}"
    )
    for problem in summary.missing:
        print(f"      note: {problem}")
    for entry in summary.cells:
        cell = entry.cell
        median = "n/a" if cell.inference.median is None else f"{cell.inference.median:.4f} s"
        fps = (
            "n/a" if cell.throughput_fps_median is None else f"{cell.throughput_fps_median:.2f} fps"
        )
        flags = "" if cell.meets_protocol else "  [below protocol minimums]"
        if not entry.consistent:
            flags += "  [machine ids differ from the directory]"
        device = ";".join(cell.devices) or "?"
        print(
            f"      {entry.machine_id:<14} {entry.suite:<26} {cell.name:<36} {device:<5} "
            f"median {median}, {fps}{flags}"
        )
    return 0


def _cmd_live_summary(directory: Path) -> int:
    """One line per report, transport and browser kept apart, plus a CSV beside them."""
    import csv

    if not directory.is_dir():
        logger.error("no live reports at %s; run unproject-live-benchmark first", directory)
        return 1
    reports, unreadable = read_reports(directory)
    if not reports and not unreadable:
        logger.error("no live reports under %s", directory)
        return 1
    for problem in unreadable:
        print(f"note: unreadable report {problem}")

    columns = [
        "file",
        "kind",
        "mode",
        "send_fps",
        "frames_sent",
        "frames_answered",
        "answered_fps",
        "dropped_by_server",
        "unaccounted",
        "round_trip_median_ms",
        "round_trip_p95_ms",
        "capture_to_present_median_ms",
        "capture_to_present_p95_ms",
        "inference_median_ms",
        "queue_wait_median_ms",
        "decode_median_ms",
        "serialize_median_ms",
        "bytes_sent",
        "bytes_received",
        "server_machine",
        "git_commit",
        "git_dirty",
    ]
    rows: list[list[str]] = []
    for report in reports:
        summary = report.summary
        # Named by its own start, so a row traces to a file.
        stamp = report.started_at.strftime("%Y%m%dT%H%M%S%f")
        rows.append(
            [
                f"{stamp}-{report.kind}-{report.mode}-{report.config.send_fps:g}fps.json",
                report.kind,
                report.mode,
                f"{report.config.send_fps:g}",
                str(summary.frames_sent),
                str(summary.frames_answered),
                _blank(summary.answered_fps),
                str(summary.frames_dropped_by_server),
                str(summary.frames_unaccounted),
                _blank(summary.round_trip_ms.median),
                _blank(summary.round_trip_ms.p95),
                _blank(summary.capture_to_present_ms.median),
                _blank(summary.capture_to_present_ms.p95),
                _blank(summary.inference_ms.median),
                _blank(summary.queue_wait_ms.median),
                _blank(summary.decode_ms.median),
                _blank(summary.serialize_ms.median),
                str(summary.bytes_sent),
                str(summary.bytes_received),
                report.environment.server_machine or "",
                report.environment.git_commit or "",
                ""
                if report.environment.git_dirty is None
                else str(report.environment.git_dirty).lower(),
            ]
        )
    target = directory / "summary.csv"
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"summarised {len(reports)} report(s) into {target}")
    for row in rows:
        present = f", capture-to-present median {row[11]} ms" if row[11] else ""
        print(
            f"      {row[1]} {row[2]} at {row[3]} fps: sent {row[4]}, answered {row[5]} "
            f"({row[6] or 'n/a'} fps), dropped {row[7]}, unaccounted {row[8]}; round trip "
            f"median {row[9] or 'n/a'} ms, p95 {row[10] or 'n/a'} ms{present}"
        )
    return 0


def _blank(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _cmd_gt_evaluate(
    config_path: Path,
    artifacts_root: Path,
    *,
    sequences: list[str] | None,
    matcher: MatcherKind | None,
    overwrite: bool,
    release_frames: bool,
    final: bool = False,
) -> int:
    """Score a ground-truth release. Validates everything before loading weights."""
    config = load_ground_truth_config(config_path)
    # An override is a different measurement, so it goes through the config and
    # lands in the evaluation identity rather than beside it. Re-validated, not
    # copied: the gate rule depends on the matcher, and `model_copy` would skip it.
    if matcher is not None:
        try:
            config = type(config).model_validate({**config.model_dump(), "matcher": matcher})
        except ValueError as error:
            raise GroundTruthError(f"--matcher {matcher}: {error}") from error
    result = evaluate(
        config,
        config_dir=config_path.resolve().parent,
        artifacts_root=artifacts_root,
        sequences=sequences,
        overwrite=overwrite,
        release_intermediate_frames=release_frames,
        final=final,
    )
    print(f"wrote {result.directory}{' [final]' if final else ' [exploratory, not final]'}")
    if result.frames_released_bytes:
        print(
            f"released {result.frames_released_bytes / 1e9:.1f} GB of decoded frames; "
            "re-run the evaluation to regenerate them"
        )
    _report(
        result.observations,
        config.thresholds,
        protocol=config.metric_protocol,
        matcher=config.matcher,
        camera=config.camera,
    )
    return 0


def _cmd_gt_summary(directory: Path, *, regenerate: bool = False) -> int:
    """Report a finished evaluation from its files alone.

    With `regenerate`, first rewrites the derived files from the observations.
    """
    if regenerate:
        descriptor, changed = regenerate_derived(directory)
        if changed:
            print(f"regenerated the derived files; these changed: {', '.join(changed)}")
        else:
            print("regenerated the derived files; every one reproduced the run byte for byte")
    descriptor, observations = read_evaluation(directory)
    print(
        f"{descriptor.config.name}: {descriptor.dataset.name} "
        f"({descriptor.dataset.release}), mapping {descriptor.mapping_identifier}"
    )
    if descriptor.dataset.excluded:
        print(
            f"note: {len(descriptor.dataset.excluded)} sequence(s) excluded: "
            f"{', '.join(descriptor.dataset.excluded)}"
        )
    if descriptor.migration is not None:
        print(
            f"note: read from schema {descriptor.migration.from_schema_version}; "
            + "; ".join(descriptor.migration.notes)
        )
    print(
        f"{'final' if descriptor.final else 'exploratory, not final'}; scored at commit "
        f"{descriptor.environment.git_commit or 'unknown'}"
        f"{' (dirty)' if descriptor.environment.git_dirty else ''}; bodies from "
        + _bodies_provenance(descriptor)
    )
    _report(
        observations,
        descriptor.config.thresholds,
        protocol=descriptor.metric_protocol,
        matcher=descriptor.matching.matcher,
        camera=descriptor.camera_source,
    )
    return 0


def _report(
    observations: Observations,
    thresholds: MetricThresholds,
    *,
    protocol: MetricProtocol,
    matcher: MatcherKind,
    camera: CameraKind,
) -> int:
    """Print the headline numbers, with the coverage they were computed over.

    Recall comes first on purpose. ADR-0004 §6.1 forbids an accuracy figure that
    travels without one, because a model that finds one easy person out of ten
    would otherwise read as excellent. The protocol comes second, and by its
    identifier: these are the project's MuPoTS-compatible numbers, not the
    official evaluator's, and every line below is read under that name.
    """
    summary = summarise_ground_truth(observations, thresholds)
    coverage = summary.coverage
    spec = PROTOCOLS[protocol]
    print(
        f"coverage: {coverage.matched} of {coverage.ground_truth_people} annotated "
        f"people scored ({summary.recall:.1%} recall) over {coverage.samples} frame(s); "
        f"{coverage.false_positives} prediction(s) unmatched"
    )
    print(
        f"protocol {spec.identifier}: {spec.common_joints} common joints, "
        f"{spec.official_pck_joints_covered} of {spec.official_pck_joints_total} official "
        f"PCK joints ({', '.join(spec.absent_official_joints)} absent), root-relative on "
        f"{spec.root_relative_field}; a project metric, not the official evaluator's score"
    )
    print(f"  matcher {matcher}, camera {camera}")
    print(
        f"  3DPCK_rel           {summary.pck_relative.micro:6.2f} % micro   "
        f"{summary.pck_relative.macro:6.2f} % macro   "
        f"({summary.pck_relative.joints} joints, {summary.pck_relative.people} people)"
    )
    print(
        f"  3DPCK_abs           {summary.pck_absolute.micro:6.2f} % micro   "
        f"{summary.pck_absolute.macro:6.2f} % macro"
    )
    print(
        f"  MPJPE               {summary.mpjpe_mm.micro:6.1f} mm       "
        f"root-relative {summary.mpjpe_relative_mm.micro:6.1f} mm"
    )
    axes = "XYZ"
    mae = "  ".join(f"{axes[i]} {summary.root_mae_xyz_mm[i]:6.1f}" for i in range(3))
    bias = "  ".join(f"{axes[i]} {summary.root_bias_xyz_mm[i]:+7.1f}" for i in range(3))
    print(f"  root MAE mm         {mae}")
    print(f"  root bias mm        {bias}")
    for depth in summary.depth:
        if depth.people == 0:
            continue
        lower = "   -inf" if depth.lower_m == float("-inf") else f"{depth.lower_m:7.1f}"
        upper = "    inf" if depth.upper_m == float("inf") else f"{depth.upper_m:7.1f}"
        print(
            f"  depth {lower} to {upper} m: {depth.people:5d} people, "
            f"bias {depth.signed_bias_mm:+8.1f} mm ({depth.relative_bias:+.1%})"
        )
    robust = robust_depth(observations.depth_m, observations.root_error_xyz_mm)
    if robust.people:
        # The mean above keeps every outlier; this says what the bulk of the
        # distribution looks like and how many there were.
        print(
            f"  depth, distribution: median {robust.median_mm:+.1f} mm, IQR "
            f"{robust.p25_mm:+.1f} to {robust.p75_mm:+.1f}, p10/p90 {robust.p10_mm:+.1f}/"
            f"{robust.p90_mm:+.1f}, |error| median {robust.abs_median_mm:.1f}, "
            f"relative median {robust.relative_median:+.1%}; outliers beyond 1 m: "
            f"{robust.gross_outliers}, beyond 3 MAD: {robust.mad_outliers} (kept in every mean)"
        )
    return 0


def _bodies_provenance(descriptor: EvaluationDescriptor) -> str:
    states = {
        "clean" if artifact.git_dirty is False else "dirty" if artifact.git_dirty else "unknown"
        for artifact in descriptor.body_artifacts
    }
    commits = sorted({artifact.git_commit or "unknown" for artifact in descriptor.body_artifacts})
    return (
        f"{len(descriptor.body_artifacts)} artifact(s), tree state {', '.join(sorted(states))}, "
        f"commit(s) {', '.join(commit[:12] for commit in commits)}"
    )


def _cmd_gt_hypotheses(directory: Path, *, out: Path | None) -> int:
    """The confirmatory test, and the post-hoc diagnostics beside it, labelled."""
    hypotheses = analyse_hypotheses(directory)
    written = write_hypotheses(hypotheses, directory, out or directory / "hypotheses")
    fit = hypotheses.fit
    print(
        f"{hypotheses.evaluation_id[:12]} ({hypotheses.camera_source} camera): "
        f"H1 {hypotheses.reading}"
    )
    print(f"  {hypotheses.hypothesis}")
    print(f"  prediction: {hypotheses.prediction}")
    if (
        fit.slope is not None
        and fit.intercept is not None
        and fit.slope_ci_low is not None
        and fit.slope_ci_high is not None
        and fit.intercept_ci_low is not None
        and fit.intercept_ci_high is not None
    ):
        print(
            f"  fit over {fit.people} people: slope {fit.slope:.3f} "
            f"[{fit.slope_ci_low:.3f}, {fit.slope_ci_high:.3f}], intercept "
            f"{fit.intercept:+.4f} [{fit.intercept_ci_low:+.4f}, {fit.intercept_ci_high:+.4f}], "
            f"R^2 {fit.r_squared or 0.0:.3f}, Spearman {fit.spearman_rho or 0.0:+.3f}; "
            f"residual |median| {fit.residual_abs_median or 0.0:.4f}"
        )
        print(
            f"  medians: observed relative error {fit.y_median or 0.0:+.4f}, predicted "
            f"{fit.p_median or 0.0:+.4f}, scale ratio {fit.ratio_median or 0.0:.4f}"
        )
    else:
        print(f"  not testable over {fit.people} people with a proxy on both sides")
    print(f"  reading rule: {hypotheses.reading_rule}")
    split = hypotheses.shape_split
    print(
        f"  post-hoc, frame shape: residual median square {split.square_residual_median} "
        f"({split.square_people} people) against wide {split.wide_residual_median} "
        f"({split.wide_people}); {split.note}"
    )
    offset = hypotheses.vertical_offset
    print(
        f"  post-hoc, vertical offset: root Y bias {offset.root_y_bias_mm} mm mean, "
        f"{offset.root_y_bias_median_mm} mm median; {offset.note}"
    )
    for path in written:
        print(f"wrote {path}")
    return 0


def _cmd_gt_compare(a: Path, b: Path, *, out: Path | None, label_a: str, label_b: str) -> int:
    """Same numbers under new labels, or the paired effect of an intervention."""
    comparison = compare_evaluations(a, b)
    text = describe(comparison)
    print(text)
    target = out or (a.parent / f"compare-{a.name}-{b.name}")
    target.mkdir(parents=True, exist_ok=True)
    (target / DIFF_FILENAME).write_text(text + "\n", encoding="utf-8")
    written = write_paired(a, b, target, label_a=label_a, label_b=label_b)
    print(f"wrote {target / DIFF_FILENAME}")
    for path in written:
        print(f"wrote {path}")
    return 0


def _cmd_rq3(run_dir: Path, annotations_path: Path, out: Path | None) -> int:
    annotation = load_annotation(annotations_path)
    run = read_run(run_dir)
    summary = analyse(run, annotation)
    directory = out or run_dir
    as_json, as_csv = write_rq3_summary(summary, directory)
    drawn = write_figures(summary, directory)

    print(
        f"{annotation.name}: {len(run.detections)} detection(s) over {len(run.track_ids)} track(s)"
    )
    print(f"wrote {as_json} and {as_csv}")
    for path in drawn:
        print(f"      figure {path}")

    for position in summary.positions:
        if position.samples == 0:
            # Silence here would read as a passing measurement.
            print(f"      {position.marker}: no detections in the annotated interval")
            continue
        print(
            f"      {position.marker}: lateral {position.lateral_abs_median_metres} m, "
            f"depth {position.depth_abs_median_metres} m "
            f"(marker known to {position.marker_uncertainty_metres} m)"
        )
    fragmentation = summary.fragmentation
    reprojection = summary.reprojection
    print(f"      {fragmentation.tracks} track(s) for {fragmentation.participants} participant(s)")
    # The clipped figure is reported too, but the clear one is the one that
    # means anything: a body cut off by the shot may have its pelvis outside it.
    print(
        f"      reprojection inside the box: "
        f"{reprojection.clear_inside_fraction} over {reprojection.clear_samples} "
        f"body/bodies fully in frame, {reprojection.clipped_inside_fraction} over "
        f"{reprojection.clipped_samples} clipped by the frame edge"
    )
    return 0


def _cmd_filter_characterise(out: Path) -> int:
    """The filter on signals whose truth is known. Smoothness and delay, never accuracy."""
    report = characterise()
    written = write_characterisation(report, out)
    print(f"{len(report.results)} signal(s) through the filter; wrote:")
    for path in written:
        print(f"      {path}")
    reference = [
        r
        for r in report.results
        if r.axis == "depth" and r.cadence_hz == 10.0 and r.distance_m == 8.0
    ]
    for row in reference:
        if row.kind == "noise":
            print(
                f"      at 8 m, 10 Hz: noise RMS {row.rms_raw_m} -> {row.rms_filtered_m} m "
                f"(x{row.attenuation_ratio})"
            )
        elif row.kind == "step":
            print(
                f"      at 8 m, 10 Hz: 1 m step reaches 90 % after {row.t90_seconds} s, "
                f"settles after {row.settling_seconds} s, overshoot {row.overshoot_fraction}"
            )
        elif row.kind == "ramp":
            print(f"      at 8 m, 10 Hz: a 1 m/s walk is followed {row.lag_seconds} s behind")
        elif row.frequency_hz is not None:
            print(
                f"      at 8 m, 10 Hz: {row.frequency_hz:g} Hz motion at {row.gain_db} dB, "
                f"delayed {row.phase_delay_seconds} s"
            )
    return 0


def _cmd_gates(root: Path, record_path: Path) -> int:
    record = run_gates(root.resolve())
    write_gates(record, record_path)
    for result in record.results:
        state = "ok" if result.exit_code == 0 else f"exit {result.exit_code}"
        print(f"      {state:>8}  {result.seconds:7.1f} s  {result.command}")
    print(f"{'green' if record.green else 'RED'}: wrote {record_path}")
    return 0 if record.green else 1


def _cmd_evidence_bundle(
    spec_path: Path, out: Path, *, root: Path, gates: Path | None, run_gates_first: bool
) -> int:
    spec = load_evidence_spec(spec_path)
    record: GatesRecord | None = None
    if run_gates_first:
        record = run_gates(root.resolve())
        print(f"gates: {'green' if record.green else 'RED'}")
    elif gates is not None:
        record = GatesRecord.model_validate_json(gates.read_text("utf-8"))
    release = build_bundle(
        spec, spec_dir=spec_path.resolve().parent, repo_root=root.resolve(), out=out, gates=record
    )
    print(
        f"{release.name}: {len(release.results)} result(s) into {out}"
        f"{' (tree dirty)' if release.git_dirty else ''}, commit "
        f"{(release.git_commit or 'unknown')[:12]}"
    )
    for result in release.results:
        carried = sum(entry.bytes for entry in result.files)
        print(
            f"      {result.status:>9}  {result.label} ({result.kind}): {len(result.files)} "
            f"file(s), {carried / 1024:.0f} KB carried, {len(result.local_only)} indexed locally"
        )
        for note in result.notes:
            print(f"                 note: {note}")
    if release.gates_green is None:
        print("      gates: not recorded; pass --run-gates or --gates before a release")
    print(f"wrote {out / RELEASE_FILENAME}")
    return 0


def _cmd_evidence_verify(bundle: Path) -> int:
    problems = verify_bundle(bundle)
    if problems:
        for problem in problems:
            print(f"      {problem}")
        print(f"{len(problems)} problem(s) in {bundle}")
        return 1
    print(f"{bundle}: every carried file matches {RELEASE_FILENAME}")
    return 0


def _collect_configs(paths: list[Path]) -> list[Path]:
    configs: list[Path] = []
    for path in paths:
        if path.is_dir():
            configs.extend(sorted(path.rglob("*.toml")))
        else:
            configs.append(path)
    return configs


def _configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
