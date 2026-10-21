#!/usr/bin/env python3
"""Check one machine's campaign results against the validation list of the
handoff note (docs/notes/2026-09-15-campaign-handoff.md §7) before they are
called thesis numbers.

    validate-campaign.py [--root artifacts/campaign] [--commit <sha>] <machine-id>...

Mechanical lines only, read from the files and never assumed: the preflight
report, every job JSON (status, frame counts, commit, dirty state, machine
id, device, suite hash), the summary (`meets_protocol`, `one_machine`), the
conditions (power on a Mac, `nvidia-smi` readings on Linux, load averages),
and which of the three suites are present. The numbers themselves are
printed beside the checks - the 16:9 → square ratio, the resolution effect,
the person-count cells, CPU against the accelerator - for a person to read;
the script does not judge them.

Exit status 0 when every hard line passes on every machine given, 1
otherwise. Stdlib only, so it runs with the system Python.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SUITES = {
    "matrix": REPO / "experiments" / "benchmark" / "v2" / "matrix-suite.toml",
    "people": REPO / "experiments" / "benchmark" / "v2" / "people-suite.toml",
    "cpu": REPO / "experiments" / "benchmark" / "v2" / "cpu-suite.toml",
}
STAGE = "benchmark-v2"


@dataclass
class Report:
    machine: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    # suite name -> commit set / suite hash, for the cross-machine lines
    commits: dict[str, set[str]] = field(default_factory=dict)
    hashes: dict[str, set[str]] = field(default_factory=dict)

    def fail(self, text: str) -> None:
        self.failures.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def say(self, text: str) -> None:
        self.lines.append(text)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a JSON object")
    return data


def registry_device(machine: str) -> str | None:
    with (REPO / "benchmarks" / "machines.toml").open("rb") as handle:
        registry = tomllib.load(handle)
    for entry in registry.get("machines", []):
        if entry.get("id") == machine:
            return str(entry.get("device"))
    return None


def check_preflight(report: Report, suite_dir: Path, suite: str) -> None:
    reports = sorted(suite_dir.glob("preflight-*.json"))
    if not reports:
        report.fail(f"{suite}: no preflight report")
        return
    latest = load_json(reports[-1])
    fails = [c for c in latest.get("checks", []) if c.get("status") == "fail"]
    warns = [c for c in latest.get("checks", []) if c.get("status") == "warn"]
    if not latest.get("ok") or fails:
        report.fail(
            f"{suite}: preflight {reports[-1].name} not ok: "
            + "; ".join(f"{c['name']}: {c['detail']}" for c in fails)
        )
    for check in warns:
        report.warn(f"{suite}: preflight warns {check['name']}: {check['detail']}")
    report.say(f"{suite}: preflight {reports[-1].name} ok={latest.get('ok')} warnings={len(warns)}")


def check_jobs(
    report: Report, suite_dir: Path, suite: str, config: dict[str, Any], expected_device: str
) -> list[dict[str, Any]]:
    cells = [c["name"] for c in config["cells"]]
    reps = int(config["repetitions"])
    warmup, measured = int(config["warmup_frames"]), int(config["measured_frames"])
    jobs: list[dict[str, Any]] = []
    for cell in cells:
        files = sorted((suite_dir / cell).glob("*-rep*-job*.json"))
        if len(files) != reps:
            report.fail(f"{suite}/{cell}: {len(files)} job files, expected {reps}")
        for path in files:
            job = load_json(path)
            jobs.append(job)
            where = f"{suite}/{cell}/{path.name}"
            env = job.get("environment", {})
            if job.get("status") != "ok":
                report.fail(f"{where}: status {job.get('status')}: {job.get('error')}")
            if job.get("dry_run"):
                report.fail(f"{where}: dry run")
            phases = Counter(f.get("phase") for f in job.get("frames", []))
            if phases.get("steady", 0) != measured or phases.get("warmup", 0) != warmup:
                report.fail(
                    f"{where}: frames warmup={phases.get('warmup')} steady={phases.get('steady')}"
                )
            if env.get("git_dirty"):
                report.fail(f"{where}: git_dirty true")
            if not env.get("git_commit"):
                report.fail(f"{where}: no git_commit")
            if env.get("machine_id") != report.machine:
                report.fail(f"{where}: machine_id {env.get('machine_id')!r}")
            if env.get("device") != expected_device:
                report.fail(f"{where}: device {env.get('device')!r}, expected {expected_device!r}")
            cond = job.get("conditions", {})
            if cond.get("load_average_before") is None or cond.get("load_average_after") is None:
                report.warn(f"{where}: load average not recorded")
            if expected_device == "mps" and cond.get("power_source") != "mains":
                report.fail(f"{where}: power_source {cond.get('power_source')!r}")
            if sys_is_linux(env) and (
                cond.get("gpu_before") is None or cond.get("gpu_after") is None
            ):
                report.fail(f"{where}: no nvidia-smi reading before/after")
            if expected_device == "cuda":
                for key in ("device_name", "cuda_version", "driver_version"):
                    if not env.get(key):
                        report.fail(f"{where}: environment.{key} empty")
                if job.get("memory", {}).get("cuda_allocated_peak_bytes") is None:
                    report.fail(f"{where}: no CUDA memory peak")
            report.commits.setdefault(suite, set()).add(str(env.get("git_commit")))
            report.hashes.setdefault(suite, set()).add(
                str(job.get("identity", {}).get("suite_hash"))
            )
    return jobs


def sys_is_linux(env: dict[str, Any]) -> bool:
    return "linux" in str(env.get("platform", "")).lower()


def check_summary(report: Report, suite_dir: Path, suite: str) -> dict[str, Any] | None:
    path = suite_dir / "summary.json"
    if not path.is_file():
        report.fail(f"{suite}: no summary.json")
        return None
    summary = load_json(path)
    for cell in summary.get("cells", []):
        if not cell.get("meets_protocol"):
            report.fail(f"{suite}/{cell['name']}: meets_protocol false: {cell.get('errors')}")
        if not cell.get("one_machine", True):
            report.fail(f"{suite}/{cell['name']}: one_machine false")
        if cell.get("load_average_max") is not None and cell["load_average_max"] > 3.0:
            report.warn(f"{suite}/{cell['name']}: load_average_max {cell['load_average_max']:.2f}")
    return summary


def describe(report: Report, suite: str, summary: dict[str, Any]) -> None:
    med: dict[str, float] = {}
    for cell in summary.get("cells", []):
        name = cell["name"]
        inference = cell["inference"].get("median")
        total = cell["total"].get("median")
        if inference is None or total is None:
            continue
        med[name] = float(inference)
        people = cell["people"].get("median")
        fps = cell.get("throughput_fps_median")
        report.say(
            f"{suite}/{name}: inference median {inference:.4f} s, total {total:.4f} s, "
            f"{fps if fps is None else f'{fps:.2f}'} fps, people {people}, "
            f"device {'/'.join(cell.get('devices', []))} {'/'.join(cell.get('device_names', []))}"
        )

    def ratio(b: str, a: str, label: str) -> None:
        if a in med and b in med and med[a] > 0:
            report.say(f"{suite}: {label}: {med[b] / med[a]:.3f} ({b} / {a})")

    ratio("mot16-09-letterbox-skeleton", "mot16-09-skeleton", "16:9 -> square, skeleton, full")
    ratio("mot16-09-letterbox-mesh", "mot16-09-mesh", "16:9 -> square, mesh, full")
    ratio(
        "mot16-09-letterbox-half-skeleton",
        "mot16-09-half-skeleton",
        "16:9 -> square, skeleton, half",
    )
    ratio("mot16-09-skeleton", "mot16-09-half-skeleton", "1920 / 960, skeleton, 16:9")
    ratio(
        "mot16-09-letterbox-skeleton",
        "mot16-09-letterbox-half-skeleton",
        "1920 / 960, skeleton, 1:1",
    )
    ratio("mot16-09-mesh", "mot16-09-skeleton", "mesh / skeleton, 16:9")
    ratio("mot16-08-skeleton", "mupots-ts10-skeleton", "high / low count, skeleton")
    ratio("mot16-09-skeleton", "mupots-ts10-skeleton", "medium / low count, skeleton")
    ratio("mupots-ts11-skeleton", "mupots-ts10-skeleton", "low / low, the two scenes")


def validate(machine: str, root: Path, commit: str | None) -> Report:
    report = Report(machine)
    device = registry_device(machine)
    if device is None:
        report.fail("machine id not in benchmarks/machines.toml")
        return report
    stage = root / machine / STAGE
    if not stage.is_dir():
        report.fail(f"no results under {stage}")
        return report
    accelerator_medians: dict[str, float] = {}
    for key, suite_path in SUITES.items():
        with suite_path.open("rb") as handle:
            config = tomllib.load(handle)
        suite = str(config["name"])
        suite_dir = stage / suite
        if not suite_dir.is_dir():
            (report.fail if key != "cpu" else report.warn)(f"{suite}: absent")
            continue
        expected = "cpu" if config.get("device") == "cpu" else device
        check_preflight(report, suite_dir, suite)
        check_jobs(report, suite_dir, suite, config, expected)
        summary = check_summary(report, suite_dir, suite)
        if summary is not None:
            describe(report, suite, summary)
            for cell in summary.get("cells", []):
                value = cell["inference"].get("median")
                if value is not None:
                    accelerator_medians[f"{key}:{cell['name']}"] = float(value)
    a, b = (
        accelerator_medians.get("cpu:mot16-09-skeleton"),
        accelerator_medians.get("matrix:mot16-09-skeleton"),
    )
    if a is not None and b is not None and b > 0:
        report.say(f"cpu / accelerator, mot16-09 skeleton: {a / b:.2f}")
    commits = set().union(*report.commits.values()) if report.commits else set()
    if len(commits) > 1:
        report.fail(f"jobs from {len(commits)} commits: {sorted(c[:12] for c in commits)}")
    if commit is not None and commits and commits != {commit}:
        report.fail(f"commit {sorted(c[:12] for c in commits)} is not the campaign's {commit[:12]}")
    for suite, hashes in report.hashes.items():
        if len(hashes) > 1:
            report.fail(f"{suite}: {len(hashes)} suite hashes among the jobs")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("machines", nargs="+", help="machine ids, as in benchmarks/machines.toml")
    parser.add_argument("--root", type=Path, default=REPO / "artifacts" / "campaign")
    parser.add_argument("--commit", help="the campaign commit every job must record (full sha)")
    args = parser.parse_args(argv)
    reports = [validate(machine, args.root, args.commit) for machine in args.machines]
    hashes: dict[str, set[str]] = {}
    for report in reports:
        print(f"== {report.machine}")
        for line in report.lines:
            print(f"   {line}")
        for line in report.warnings:
            print(f"   WARN {line}")
        for line in report.failures:
            print(f"   FAIL {line}")
        for suite, values in report.commits.items():
            print(f"   {suite}: commit {', '.join(sorted(c[:12] for c in values))}")
        for suite, values in report.hashes.items():
            hashes.setdefault(suite, set()).update(values)
        verdict = (
            "valid" if not report.failures else f"PILOT ({len(report.failures)} failed line(s))"
        )
        print(f"   verdict: {verdict}")
    for suite, values in hashes.items():
        if len(values) > 1:
            print(f"FAIL {suite}: suite hash differs across machines: {sorted(values)}")
            return 1
    return 0 if all(not report.failures for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
