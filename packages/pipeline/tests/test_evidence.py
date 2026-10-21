"""Tests for the evidence bundle and the gate list. ADR-0005 WP7.

The bundle is built from synthetic results - an evaluation, a benchmark suite,
a filter report, a bodies run - so what is checked is the policy: only
allowed files are carried, none over the limit, a non-final evaluation is
refused where the spec demands final, everything local is indexed by hash,
and a bundle on disk can be verified against its own manifest.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from test_benchmark_suite import _environment, _result
from test_ground_truth_report import _config, _evaluation_dir, _observations
from unproject.evidence import (
    RELEASE_FILENAME,
    EvidenceError,
    EvidenceSpec,
    Release,
    build_bundle,
    load_evidence_spec,
    verify_bundle,
)
from unproject.gates import GATES, GateResult, GatesRecord, ci_run_lines, run_gates
from unproject.rq3.characterise import characterise, write_characterisation

_ROOT = Path(__file__).resolve().parents[3]


def _spec(results: list[dict[str, object]], **extra: object) -> EvidenceSpec:
    return EvidenceSpec.model_validate({"name": "thesis", "results": results} | extra)


class TestTheGateList:
    def test_it_is_exactly_what_ci_runs(self) -> None:
        """One list, two places to run it. If they diverge, "green" stops
        meaning the same thing on the runner and in a bundle."""
        lines = ci_run_lines(_ROOT / ".github" / "workflows" / "ci.yml")

        assert lines == [" ".join(argv) for argv in GATES]

    def test_the_shipped_spec_loads_and_names_every_kind(self) -> None:
        spec = load_evidence_spec(_ROOT / "experiments" / "evidence" / "thesis.toml")

        assert {r.kind for r in spec.results} == {
            "benchmark-suite",
            "live",
            "ground-truth",
            "compare",
            "filter",
            "rq3-run",
        }
        assert all(
            r.require_final
            for r in spec.results
            if r.kind in {"ground-truth", "benchmark-suite", "rq3-run"}
        )

    def test_running_a_gate_records_its_exit_and_output(self, tmp_path: Path) -> None:
        record = run_gates(tmp_path, commands=(("true",), ("false",)))

        assert [r.exit_code for r in record.results] == [0, 1]
        assert record.green is False
        assert GatesRecord.model_validate_json(record.model_dump_json()) == record


class TestBuildingABundle:
    def test_it_carries_the_allowed_files_and_indexes_the_rest(self, tmp_path: Path) -> None:
        directory = _evaluation_dir(tmp_path / "artifacts", "abcd", _observations([10.0, -20.0]))
        write_characterisation(characterise(), tmp_path / "artifacts" / "filter")
        spec = _spec(
            [
                {
                    "kind": "ground-truth",
                    "label": "control",
                    "path": "artifacts/abcd",
                    "command": "uv run unproject gt-evaluate x.toml",
                },
                {
                    "kind": "filter",
                    "label": "filter",
                    "path": "artifacts/filter",
                    "command": "uv run unproject filter-characterise",
                },
            ]
        )

        release = build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")

        control = release.results[0]
        assert control.status == "collected"
        carried = {entry.path for entry in control.files}
        assert "control/summary.json" in carried
        assert "control/evaluation.json" in carried
        assert "control/per-sequence.csv" in carried
        assert not any(path.endswith("observations.npz") for path in carried)
        assert [entry.path for entry in control.local_only] == ["artifacts/abcd/observations.npz"]
        assert control.identity["evaluation_id"] == "abcd" * 8
        assert control.identity["final"] is False
        assert "exploratory evaluation, not final" in control.notes
        assert (tmp_path / "bundle" / "filter" / "filter-characterisation.json").is_file()
        assert (tmp_path / "bundle" / RELEASE_FILENAME).is_file()
        assert verify_bundle(tmp_path / "bundle") == []
        assert directory.is_dir()

    def test_a_non_final_evaluation_is_refused_where_final_is_required(
        self, tmp_path: Path
    ) -> None:
        _evaluation_dir(tmp_path / "artifacts", "abcd", _observations([10.0]))
        spec = _spec(
            [
                {
                    "kind": "ground-truth",
                    "label": "control",
                    "path": "artifacts/abcd",
                    "command": "x",
                    "require_final": True,
                }
            ]
        )

        with pytest.raises(EvidenceError, match="exploratory"):
            build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")

    def test_a_missing_optional_result_is_recorded_as_absent(self, tmp_path: Path) -> None:
        spec = _spec(
            [
                {
                    "kind": "live",
                    "label": "live",
                    "path": "artifacts/live",
                    "command": "x",
                    "optional": True,
                }
            ]
        )

        release = build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")

        assert release.results[0].status == "absent"
        assert release.results[0].notes == ["not yet produced"]

    def test_a_missing_required_result_names_the_command(self, tmp_path: Path) -> None:
        spec = _spec(
            [{"kind": "live", "label": "live", "path": "artifacts/live", "command": "run me"}]
        )

        with pytest.raises(EvidenceError, match="run me"):
            build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")

    def test_a_suite_is_a_thesis_result_only_from_one_clean_commit(self, tmp_path: Path) -> None:
        """Every job records its commit and its dirty state. A suite with a
        dirty job, or jobs from two commits, is not what the thesis quotes."""
        suite = tmp_path / "artifacts" / "suite"
        suite.mkdir(parents=True)
        clean = _result([0.1, 0.1], job_index=0)
        (suite / "rep0-job000.json").write_text(clean.model_dump_json(), encoding="utf-8")
        spec = _spec(
            [
                {
                    "kind": "benchmark-suite",
                    "label": "rq1rq2-v2",
                    "path": "artifacts/suite",
                    "command": "x",
                    "require_final": True,
                }
            ]
        )

        release = build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "b")
        [record] = release.results
        assert record.identity["jobs"] == 1
        assert record.identity["git_dirty"] is False
        assert record.identity["git_commits"] == "0" * 40

        dirty = _result(
            [0.1, 0.1],
            job_index=1,
            environment=_environment().model_copy(update={"git_dirty": True}),
        )
        (suite / "rep0-job001.json").write_text(dirty.model_dump_json(), encoding="utf-8")
        with pytest.raises(EvidenceError, match="1 of 2 job\\(s\\) ran from a dirty tree"):
            build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "b")

        elsewhere = _result(
            [0.1, 0.1],
            job_index=1,
            environment=_environment().model_copy(update={"git_commit": "1" * 40}),
        )
        (suite / "rep0-job001.json").write_text(elsewhere.model_dump_json(), encoding="utf-8")
        with pytest.raises(EvidenceError, match="jobs from 2 commit"):
            build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "b")

        # Without the requirement the same suite is carried, and its identity
        # lists both commits.
        lax = _spec([spec.results[0].model_dump() | {"require_final": False}])
        release = build_bundle(lax, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "b")
        assert release.results[0].status == "collected"
        assert release.results[0].identity["git_commits"] == ",".join(["0" * 40, "1" * 40])

    def test_a_file_over_the_limit_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "artifacts" / "live").mkdir(parents=True)
        (tmp_path / "artifacts" / "live" / "huge.json").write_bytes(b"0" * (2 * 1024 * 1024 + 1))
        spec = _spec([{"kind": "live", "label": "live", "path": "artifacts/live", "command": "x"}])

        with pytest.raises(EvidenceError, match="over the"):
            build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")

    def test_inputs_are_indexed_by_hash_and_never_copied(self, tmp_path: Path) -> None:
        weights = tmp_path / "models" / "multihmr2.pt"
        weights.parent.mkdir()
        weights.write_bytes(b"not really weights")
        write_characterisation(characterise(), tmp_path / "artifacts" / "filter")
        spec = _spec(
            [{"kind": "filter", "label": "filter", "path": "artifacts/filter", "command": "x"}],
            inputs=["models/multihmr2.pt", "models/absent.pt"],
        )

        release = build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")

        [entry] = release.inputs
        assert entry.path == "models/multihmr2.pt"
        assert entry.bytes == 18
        assert not (tmp_path / "bundle" / "models").exists()

    def test_the_gates_record_is_carried(self, tmp_path: Path) -> None:
        write_characterisation(characterise(), tmp_path / "artifacts" / "filter")
        spec = _spec(
            [{"kind": "filter", "label": "filter", "path": "artifacts/filter", "command": "x"}]
        )
        gates = GatesRecord(
            started_at=datetime(2026, 9, 15, 10, 0, tzinfo=UTC),
            finished_at=datetime(2026, 9, 15, 10, 1, tzinfo=UTC),
            git_commit="a" * 40,
            git_dirty=False,
            green=True,
            results=[GateResult(command="true", exit_code=0, seconds=0.1, output_tail="")],
        )

        release = build_bundle(
            spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle", gates=gates
        )

        assert release.gates_green is True
        assert (tmp_path / "bundle" / "gates.json").is_file()

    def test_a_rebuild_replaces_the_bundle_whole(self, tmp_path: Path) -> None:
        write_characterisation(characterise(), tmp_path / "artifacts" / "filter")
        spec = _spec(
            [{"kind": "filter", "label": "filter", "path": "artifacts/filter", "command": "x"}]
        )
        out = tmp_path / "bundle"
        out.mkdir()
        (out / "stale.txt").write_text("old", encoding="utf-8")

        build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=out)

        assert not (out / "stale.txt").exists()

    def test_a_ground_truth_bundle_carries_its_hypotheses_when_present(
        self, tmp_path: Path
    ) -> None:
        from unproject.ground_truth.hypotheses import analyse, write_hypotheses

        directory = _evaluation_dir(
            tmp_path / "artifacts",
            "abcd",
            _observations([10.0, -20.0, 5.0, 8.0]),
            config=_config(camera="annotation_derived"),
        )
        write_hypotheses(analyse(directory), directory, directory / "hypotheses")
        spec = _spec(
            [{"kind": "ground-truth", "label": "t", "path": "artifacts/abcd", "command": "x"}]
        )

        release = build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")

        carried = {entry.path for entry in release.results[0].files}
        assert "t/hypotheses/hypotheses.json" in carried
        assert "t/hypotheses/scale-ratio-vs-depth-error.svg" in carried


class TestVerifying:
    def test_a_moved_hash_is_named(self, tmp_path: Path) -> None:
        write_characterisation(characterise(), tmp_path / "artifacts" / "filter")
        spec = _spec(
            [{"kind": "filter", "label": "filter", "path": "artifacts/filter", "command": "x"}]
        )
        out = tmp_path / "bundle"
        build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=out)
        (out / "filter" / "filter-characterisation.csv").write_text("edited", encoding="utf-8")
        (out / "filter" / "filter-attenuation.svg").unlink()

        problems = verify_bundle(out)

        assert sorted(problems) == [
            "filter/filter-attenuation.svg: missing",
            "filter/filter-characterisation.csv: sha256 differs from release.json",
        ]

    def test_a_bundle_without_a_manifest_is_one_problem(self, tmp_path: Path) -> None:
        assert len(verify_bundle(tmp_path)) == 1

    def test_the_release_reads_back(self, tmp_path: Path) -> None:
        write_characterisation(characterise(), tmp_path / "artifacts" / "filter")
        spec = _spec(
            [{"kind": "filter", "label": "filter", "path": "artifacts/filter", "command": "x"}]
        )
        release = build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")

        stored = Release.model_validate_json(
            (tmp_path / "bundle" / RELEASE_FILENAME).read_text("utf-8")
        )
        assert stored == release
        assert (
            json.loads((tmp_path / "bundle" / RELEASE_FILENAME).read_text("utf-8"))["name"]
            == "thesis"
        )


def _bodies_run(directory: Path, *, frames: int = 6, bone_count: int = 2) -> Path:
    """A small bodies artifact on disk: one track walking away, plus a second
    short one, so the longest track is unambiguous."""
    import numpy as np

    from unproject.bodies.artifacts import BodiesArtifact

    directory.mkdir(parents=True)
    people: list[tuple[int, int, float]] = []  # (frame, track, depth)
    for index in range(frames):
        people.append((index, 7, 8.0 + 0.5 * index))
        if index < 2:
            people.append((index, 3, 4.0))
    count = len(people)
    root = np.array([[0.0, 0.0, depth] for _, _, depth in people], dtype="<f4")
    shape = np.zeros((count, 11), dtype="<f4")
    joints = np.zeros((count, bone_count, 3), dtype="<f4")
    joints[:, :, 2] = root[:, 2:3]
    blocks = [("root_translation", root), ("shape", shape), ("joints_3d", joints)]
    arrays = []
    offset = 0
    payload = b""
    for name, block in blocks:
        raw = block.tobytes()
        arrays.append(
            {
                "name": name,
                "shape": list(block.shape),
                "byte_offset": offset,
                "byte_length": len(raw),
            }
        )
        offset += len(raw)
        payload += raw
    (directory / "bodies.bin").write_bytes(payload)
    frame_records = []
    row = 0
    for index in range(frames):
        in_frame = [(f, t, d) for f, t, d in people if f == index]
        frame_records.append(
            {
                "frame_index": index,
                "source_index": index,
                "timestamp_seconds": index * 0.1,
                "camera": {
                    "width": 1280,
                    "height": 720,
                    "fx": 1000.0,
                    "fy": 1000.0,
                    "cx": 640.0,
                    "cy": 360.0,
                    "source": "estimated",
                },
                "people": [
                    {
                        "parameter_index": row + position,
                        "track_id": track,
                        "confidence": 0.9,
                        "box_xyxy": [0.0, 0.0, 10.0, 10.0],
                    }
                    for position, (_, track, _) in enumerate(in_frame)
                ],
            }
        )
        row += len(in_frame)
    artifact = BodiesArtifact.model_validate(
        {
            "convention": "camera_relative_opencv_metres",
            "ingest_hash": "0" * 12,
            "body_model": {
                "name": "anny",
                "version": "0.6.0",
                "bone_count": bone_count,
                "shape_count": 11,
                "bone_parents": [-1, 0],
            },
            "estimator": {
                "name": "multihmr2",
                "checkpoint_sha256": "f" * 64,
                "backbone_repo": "facebookresearch/dinov3",
                "backbone_ref": "0" * 40,
                "config_hash": "deadbeef",
            },
            "parameters": {"path": "bodies.bin", "person_count": count, "arrays": arrays},
            "frames": frame_records,
            "summary": {
                "frames_total": frames,
                "frames_with_people": frames,
                "people_total": count,
                "max_people_in_frame": 2,
                "confidence_threshold": 0.4,
                "track_count": 2,
            },
        }
    )
    (directory / "bodies.json").write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps({"git_commit": "b" * 40, "git_dirty": False}), encoding="utf-8"
    )
    return directory


class TestTheRq3RunKind:
    def test_it_derives_the_track_depth_and_the_middle_frame_and_indexes_the_run(
        self, tmp_path: Path
    ) -> None:
        run = _bodies_run(tmp_path / "artifacts" / "pedestrians" / "bodies" / "abc")
        spec = _spec(
            [
                {
                    "kind": "rq3-run",
                    "label": "pedestrians-run",
                    "path": "artifacts/pedestrians/bodies/abc",
                    "command": "uv run unproject bodies experiments/pedestrians.toml",
                }
            ]
        )

        release = build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")

        [record] = release.results
        carried = {entry.path for entry in record.files}
        assert "pedestrians-run/track-depth.csv" in carried
        assert "pedestrians-run/frame-000003.json" in carried
        assert not any(path.endswith("bodies.bin") for path in carried)
        assert {entry.path for entry in record.local_only} == {
            "artifacts/pedestrians/bodies/abc/bodies.json",
            "artifacts/pedestrians/bodies/abc/bodies.bin",
            "artifacts/pedestrians/bodies/abc/manifest.json",
        }
        assert record.identity["git_dirty"] is False
        rows = (
            (tmp_path / "bundle" / "pedestrians-run" / "track-depth.csv").read_text().splitlines()
        )
        # The longest track, 7, over its six frames; raw depth as written.
        assert rows[0] == "track_id,frame_index,time_seconds,raw_z_metres,filtered_z_metres"
        assert len(rows) == 7
        assert rows[1].split(",")[:2] == ["7", "0"]
        assert float(rows[1].split(",")[3]) == 8.0
        frame = json.loads(
            (tmp_path / "bundle" / "pedestrians-run" / "frame-000003.json").read_text()
        )
        assert frame["frame_index"] == 3
        assert frame["bone_parents"] == [-1, 0]
        assert len(frame["people"]) == 1
        assert frame["people"][0]["joints_3d"][0][2] == pytest.approx(9.5)
        assert run.is_dir()

    def test_a_run_from_a_dirty_or_unrecorded_tree_is_not_a_thesis_result(
        self, tmp_path: Path
    ) -> None:
        run = _bodies_run(tmp_path / "artifacts" / "pedestrians" / "bodies" / "abc")
        spec = _spec(
            [
                {
                    "kind": "rq3-run",
                    "label": "pedestrians-run",
                    "path": "artifacts/pedestrians/bodies/abc",
                    "command": "x",
                    "require_final": True,
                }
            ]
        )
        build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")

        (run / "manifest.json").write_text(
            json.dumps({"git_commit": "b" * 40, "git_dirty": True}), encoding="utf-8"
        )
        with pytest.raises(EvidenceError, match="dirty or unrecorded"):
            build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")

        (run / "manifest.json").unlink()
        with pytest.raises(EvidenceError, match="dirty or unrecorded"):
            build_bundle(spec, spec_dir=tmp_path, repo_root=tmp_path, out=tmp_path / "bundle")
