"""Tests for the `unproject` command line interface."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from fixture_types import MakeImageDir, MakePanningVideo, MakeVideo, WriteConfig
from unproject.cli import main

_CONFIG = """
name = "fixture"
seed = 7

[ingest]
source_path = "clip.avi"
frame_stride = 2
"""


def test_validate_config_accepts_a_valid_file(
    write_config: WriteConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(_CONFIG)

    assert main(["validate-config", str(path)]) == 0
    assert "ok" in capsys.readouterr().out


def test_validate_config_walks_a_directory(
    write_config: WriteConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_config(_CONFIG, name="one.toml")
    write_config(_CONFIG.replace('"fixture"', '"other"'), name="two.toml")

    assert main(["validate-config", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("ok") == 2


def test_validate_config_fails_on_a_bad_file(write_config: WriteConfig) -> None:
    path = write_config("name = 'fixture'\n[ingest]\nsource_path = 'clip.avi'\nnope = 1\n")

    assert main(["validate-config", str(path)]) == 1


def test_validate_config_reports_an_empty_directory(tmp_path: Path) -> None:
    assert main(["validate-config", str(tmp_path)]) == 1


def test_ingest_runs_end_to_end(
    make_video: MakeVideo,
    write_config: WriteConfig,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_video(name="clip.avi", frame_count=6)
    config = write_config(_CONFIG)
    artifacts = tmp_path / "artifacts"

    exit_code = main(["ingest", str(config), "--artifacts-root", str(artifacts)])

    assert exit_code == 0
    assert "wrote 3 frame(s)" in capsys.readouterr().out
    assert list(artifacts.rglob("ingest.json"))


def test_ingest_reports_a_missing_video(write_config: WriteConfig, tmp_path: Path) -> None:
    config = write_config(_CONFIG)

    assert main(["ingest", str(config), "--artifacts-root", str(tmp_path / "artifacts")]) == 1


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert "unproject" in capsys.readouterr().out


_IMAGE_CONFIG = """
name = "fixture"
seed = 7

[ingest]
source_path = "frames"
image_fps = 10.0
"""


def test_ingest_accepts_a_directory_of_images(
    make_image_dir: MakeImageDir,
    write_config: WriteConfig,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_image_dir(name="frames", frame_count=4)
    config = write_config(_IMAGE_CONFIG)
    artifacts = tmp_path / "artifacts"

    exit_code = main(["ingest", str(config), "--artifacts-root", str(artifacts)])

    assert exit_code == 0
    assert "wrote 4 frame(s)" in capsys.readouterr().out


def test_ingest_reports_a_directory_without_image_fps(
    make_image_dir: MakeImageDir, write_config: WriteConfig, tmp_path: Path
) -> None:
    make_image_dir(name="frames", frame_count=4)
    config = write_config(_IMAGE_CONFIG.replace("image_fps = 10.0\n", ""))

    assert main(["ingest", str(config), "--artifacts-root", str(tmp_path / "artifacts")]) == 1


_PREPROCESS_CONFIG = """
name = "fixture"
seed = 7

[ingest]
source_path = "pan.avi"

[preprocess]
min_disparity_px = 10.0
blur_percentile = 0.0
first_frame_window = 1
"""


def test_preprocess_runs_the_whole_chain(
    make_panning_video: MakePanningVideo,
    write_config: WriteConfig,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_panning_video(name="pan.avi", frame_count=13, step_px=4)
    config = write_config(_PREPROCESS_CONFIG)
    artifacts = tmp_path / "artifacts"

    exit_code = main(["preprocess", str(config), "--artifacts-root", str(artifacts)])

    assert exit_code == 0
    # 4 px of pan per frame against a 10 px threshold gives every third frame.
    assert "kept 5 of 13 frame(s)" in capsys.readouterr().out
    assert list(artifacts.rglob("preprocess.json"))
    # Stage [1] ran too.
    assert list(artifacts.rglob("ingest.json"))


def test_preprocess_reports_an_impossible_threshold(
    make_panning_video: MakePanningVideo, write_config: WriteConfig, tmp_path: Path
) -> None:
    make_panning_video(name="pan.avi", frame_count=8)
    config = write_config(
        _PREPROCESS_CONFIG.replace("min_disparity_px = 10.0", "min_disparity_px = 5000.0")
    )

    assert main(["preprocess", str(config), "--artifacts-root", str(tmp_path / "artifacts")]) == 1


def test_preprocess_rejects_a_calibration_path(write_config: WriteConfig, tmp_path: Path) -> None:
    config = write_config(_PREPROCESS_CONFIG + 'calibration_path = "camera.json"\n')

    # The field has to be rejected rather than quietly ignored.
    assert main(["validate-config", str(config)]) == 1


def test_pose_says_what_is_missing_without_the_learned_backend(
    make_panning_video: MakePanningVideo, write_config: WriteConfig, tmp_path: Path
) -> None:
    if importlib.util.find_spec("lingbot_map") is not None:
        pytest.skip("the lingbot extra is installed, so this path cannot be exercised")

    make_panning_video(name="pan.avi", frame_count=13, step_px=4)
    config = write_config(_PREPROCESS_CONFIG + '\n[pose]\nbackend = "lingbot"\n')

    # The extra is a multi-gigabyte install, so asking for the backend without it
    # has to name what is missing rather than fall back to the other backend.
    assert main(["pose", str(config), "--artifacts-root", str(tmp_path / "artifacts")]) == 1


_BENCHMARK = """
name = "fixture-skeleton"
experiment = "one.toml"
warmup_frames = 1
measured_frames = 2
repetitions = 1
"""


def test_validate_config_reads_both_kinds_in_one_pass(
    write_config: WriteConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`experiments/` holds experiments and benchmarks; the gate covers both."""
    write_config(_CONFIG, name="one.toml")
    write_config(_BENCHMARK, name="bench.toml")

    assert main(["validate-config", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "experiment=fixture" in out
    assert "benchmark=fixture-skeleton" in out


def test_validate_config_fails_on_a_bad_benchmark(write_config: WriteConfig) -> None:
    path = write_config(_BENCHMARK + "nope = 1\n", name="bench.toml")

    assert main(["validate-config", str(path)]) == 1


def test_benchmark_dry_run_writes_a_result(
    make_video: MakeVideo,
    write_config: WriteConfig,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_video(name="clip.avi", frame_count=6)
    write_config(_CONFIG.replace("frame_stride = 2", "frame_stride = 1"), name="one.toml")
    bench = write_config(_BENCHMARK, name="bench.toml")
    artifacts = tmp_path / "artifacts"

    code = main(["benchmark", str(bench), "--artifacts-root", str(artifacts), "--dry-run"])

    out = capsys.readouterr().out
    assert code == 0
    # Loud about measuring nothing: the result file looks like any other.
    assert "dry run" in out
    assert list((artifacts / "benchmark" / "fixture-skeleton").glob("*.json"))


def test_benchmark_summary_runs_no_inference_and_excludes_dry_runs(
    make_video: MakeVideo,
    write_config: WriteConfig,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_video(name="clip.avi", frame_count=6)
    write_config(_CONFIG.replace("frame_stride = 2", "frame_stride = 1"), name="one.toml")
    bench = write_config(_BENCHMARK, name="bench.toml")
    artifacts = tmp_path / "artifacts"
    main(["benchmark", str(bench), "--artifacts-root", str(artifacts), "--dry-run"])
    capsys.readouterr()

    code = main(["benchmark-summary", str(artifacts / "benchmark")])

    out = capsys.readouterr().out
    assert code == 0
    assert "excluded 1 dry run" in out
    assert (artifacts / "benchmark" / "summary.json").is_file()
    assert (artifacts / "benchmark" / "summary.csv").is_file()


def test_benchmark_summary_reports_an_empty_directory(tmp_path: Path) -> None:
    assert main(["benchmark-summary", str(tmp_path / "nothing")]) == 1


_SUITE = """
name = "smoke"
warmup_frames = 1
measured_frames = 3
repetitions = 1
[[cells]]
name = "fixture-skeleton"
experiment = "one.toml"
[[cells]]
name = "fixture-mesh"
experiment = "one.toml"
output = "mesh"
[[pairs]]
a = "fixture-skeleton"
b = "fixture-mesh"
question = "output"
"""


def test_validate_config_reads_benchmark_suites(
    write_config: WriteConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_config(_CONFIG, name="one.toml")
    write_config(_SUITE, name="suite.toml")

    assert main(["validate-config", str(tmp_path)]) == 0
    assert "benchmark-suite=smoke, 2 cell(s) x 1 repetition(s)" in capsys.readouterr().out


def test_benchmark_suite_dry_run_then_summary(
    make_video: MakeVideo,
    write_config: WriteConfig,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Protocol v2 end to end through the CLI: one process per job, then a
    summary that runs no inference and says the dry runs measured nothing."""
    make_video(name="clip.avi", frame_count=6)
    write_config(_CONFIG.replace("frame_stride = 2", "frame_stride = 1"), name="one.toml")
    suite = write_config(_SUITE, name="suite.toml")
    artifacts = tmp_path / "artifacts"

    code = main(["benchmark-suite", str(suite), "--artifacts-root", str(artifacts), "--dry-run"])

    out = capsys.readouterr().out
    assert code == 0
    assert "dry run" in out
    assert "2 job(s)" in out
    results = artifacts / "benchmark-v2" / "smoke"
    assert len(list(results.rglob("*-job*.json"))) == 2
    assert list(results.glob("*suite-run.json"))

    code = main(["benchmark-suite-summary", str(suite), "--artifacts-root", str(artifacts)])

    out = capsys.readouterr().out
    assert code == 0
    assert "excluded 2 dry run" in out
    assert (results / "summary.json").is_file()
    assert (results / "pairs.csv").is_file()


def test_benchmark_suite_summary_reports_a_missing_directory(
    write_config: WriteConfig, tmp_path: Path
) -> None:
    suite = write_config(_SUITE, name="suite.toml")
    assert main(["benchmark-suite-summary", str(suite), "--artifacts-root", str(tmp_path)]) == 1


_ANNOTATION = """
name = "corridor"
[scene]
participants = 1
[[markers]]
id = "A"
x_metres = 0.0
z_metres = 5.0
uncertainty_metres = 0.05
[[stationary]]
marker = "A"
start_seconds = 0.0
end_seconds = 1.0
"""


def test_validate_config_reads_annotations_too(
    write_config: WriteConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Experiments, benchmarks and annotations share experiments/; the gate covers all three."""
    write_config(_CONFIG, name="one.toml")
    write_config(_BENCHMARK, name="bench.toml")
    write_config(_ANNOTATION, name="corridor.toml")

    assert main(["validate-config", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "annotations=corridor" in out


def test_validate_config_fails_on_broken_annotations(write_config: WriteConfig) -> None:
    path = write_config(_ANNOTATION.replace('marker = "A"', 'marker = "Z"'), name="bad.toml")

    assert main(["validate-config", str(path)]) == 1


def test_validate_config_reads_annotations_with_no_markers(
    write_config: WriteConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare `markers = []` written after a table header belongs to that table.

    Discriminating on it alone read such a file as an experiment and failed with
    a confusing error about a missing ingest section.
    """
    path = write_config(
        'convention = "camera_relative_opencv_metres"\n'
        'name = "empty"\nmarkers = []\n[scene]\nparticipants = 1\n',
        name="empty.toml",
    )

    assert main(["validate-config", str(path)]) == 0
    assert "annotations=empty" in capsys.readouterr().out


_GROUND_TRUTH = """
schema_version = 2
name = "mupots-test"
dataset = "mupots-3d"
dataset_root = "../datasets/mupots-3d"
release = "downloaded 2026-09-08"
experiment = "one.toml"
mapping = "m.toml"
"""


def test_validate_config_reads_ground_truth_configs(
    write_config: WriteConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ground-truth config also names an `experiment`, so the benchmark branch
    would swallow it if the checks were in the other order."""
    write_config(_CONFIG, name="one.toml")
    write_config(_BENCHMARK, name="bench.toml")
    write_config(_GROUND_TRUTH, name="gt.toml")

    assert main(["validate-config", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ground-truth=mupots-test" in out
    assert "benchmark=fixture-skeleton" in out


def test_validate_config_fails_on_a_broken_ground_truth_config(
    write_config: WriteConfig,
) -> None:
    path = write_config(_GROUND_TRUTH + "nope = 1\n", name="gt.toml")

    assert main(["validate-config", str(path)]) == 1


_MAPPING = """
schema_version = 1
identifier = "test-v1"
anny_package_version = "0.6.0"
anny_bone_count = 163
checkpoint_sha256 = "fd2d9ab7010a5f590a3db2480f5f476fd3ec6afbbde80babbec0d821ba9763d6"
[[joints]]
name = "pelvis"
provenance = "both releases name a pelvis"
mupots = { names = ["pelvis"] }
anny = { names = ["root"] }
"""


def test_validate_config_reads_joint_mappings(
    write_config: WriteConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fifth kind of TOML lives under experiments/; the gate has to cover it."""
    write_config(_CONFIG, name="one.toml")
    write_config(_MAPPING, name="m.toml")

    assert main(["validate-config", str(tmp_path)]) == 0
    assert "mapping=test-v1, 1 common joint(s)" in capsys.readouterr().out


def test_validate_config_fails_on_a_broken_mapping(write_config: WriteConfig) -> None:
    path = write_config(_MAPPING.replace('name = "pelvis"', 'name = ""'), name="m.toml")

    assert main(["validate-config", str(path)]) == 1


_EVIDENCE = """
schema_version = 1
name = "thesis"
inputs = ["../models/multihmr2.pt"]
[[results]]
kind = "filter"
label = "filter"
path = "../artifacts/filter"
command = "uv run unproject filter-characterise"
optional = true
"""


def test_validate_config_reads_evidence_specs(
    write_config: WriteConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The seventh kind under experiments/. `unproject gates` from a fresh clone
    is what found the gate rejecting the shipped spec as a broken experiment."""
    write_config(_CONFIG, name="one.toml")
    write_config(_EVIDENCE, name="thesis.toml")

    assert main(["validate-config", str(tmp_path)]) == 0
    assert "evidence=thesis, 1 result(s), 1 optional, 1 input(s) indexed" in capsys.readouterr().out


def test_validate_config_fails_on_a_broken_evidence_spec(write_config: WriteConfig) -> None:
    path = write_config(_EVIDENCE.replace('kind = "filter"', 'kind = "video"'), name="e.toml")

    assert main(["validate-config", str(path)]) == 1


def test_gt_summary_regenerates_and_gt_compare_reports_identical_numbers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR-0005 WP4 through the CLI: the derived files reproduce, the report
    names the protocol and the tree state, and two evaluations with the same
    numbers under different labels compare as identical."""
    from test_ground_truth_report import _config, _evaluation_dir, _observations

    observations = _observations([10.0, -20.0, 30.0])
    a = _evaluation_dir(tmp_path, "aaaa", observations)
    b = _evaluation_dir(tmp_path, "bbbb", observations, config=_config(camera="annotation_derived"))

    assert main(["gt-summary", str(a), "--regenerate"]) == 0
    out = capsys.readouterr().out
    assert "reproduced the run byte for byte" in out
    assert "protocol unproject_mupots_14j_v1" in out
    assert "exploratory, not final" in out
    assert "depth, distribution: median" in out

    assert main(["gt-compare", str(a), str(b), "--out", str(tmp_path / "cmp")]) == 0
    out = capsys.readouterr().out
    assert "numbers: identical" in out
    assert "camera_source: 'model_estimated' -> 'annotation_derived'" in out
    assert (tmp_path / "cmp" / "paired-per-sequence.csv").is_file()
    assert (tmp_path / "cmp" / "paired-depth-bias.svg").is_file()
    assert (tmp_path / "cmp" / "descriptor-diff.txt").is_file()


def test_gt_hypotheses_runs_the_pre_registered_test_and_labels_the_rest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from test_ground_truth_hypotheses import _scaled_observations
    from test_ground_truth_report import _config, _evaluation_dir

    ratios = [1.0 + 0.01 * (index - 10) for index in range(21)]
    observations = _scaled_observations(ratios, [r - 1.0 for r in ratios])
    directory = _evaluation_dir(
        tmp_path, "aaaa", observations, config=_config(camera="annotation_derived")
    )

    assert main(["gt-hypotheses", str(directory)]) == 0

    out = capsys.readouterr().out
    assert "H1 supported" in out
    assert "reading rule:" in out and "declared before the data" in out
    assert "post-hoc, frame shape" in out
    assert "post-hoc, vertical offset" in out
    assert (directory / "hypotheses" / "hypotheses.json").is_file()
    assert (directory / "hypotheses" / "scale-ratio-vs-depth-error.svg").is_file()


def test_filter_characterise_writes_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["filter-characterise", "--out", str(tmp_path / "filter")]) == 0

    out = capsys.readouterr().out
    assert "signal(s) through the filter" in out
    assert "1 m step reaches 90 %" in out
    assert (tmp_path / "filter" / "filter-characterisation.json").is_file()
    assert (tmp_path / "filter" / "filter-attenuation.svg").is_file()
