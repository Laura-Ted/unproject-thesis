"""Tests for the multi-machine campaign of ADR-0006.

The model never runs here. Under test: that the device a suite asks for is
the device a job gets or an error, never a substitute; that a result says
which declared machine and which device it ran on and the summary refuses to
pool two; that the Linux conditions parse; that a campaign folds one machine's
summaries per directory; and that the preflight finds what it is there to
find without touching anything.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fixture_types import MakeVideo, WriteConfig
from test_benchmark_suite import _result
from unproject.benchmark.artifacts import ENV_MACHINE_ID, Environment
from unproject.benchmark.campaign import (
    CAMPAIGN_CSV_FILENAME,
    CAMPAIGN_FILENAME,
    MachineRegistry,
    load_registry,
    machine_root,
    read_campaign,
    to_csv,
    write_campaign,
)
from unproject.benchmark.job import apply_device_policy, parse_conditions, parse_nvidia_smi
from unproject.benchmark.preflight import (
    PREFLIGHT_PREFIX,
    Check,
    PreflightReport,
    run_preflight,
    write_report,
)
from unproject.benchmark.results import AcceleratorReading, Conditions
from unproject.benchmark.run import BenchmarkError
from unproject.benchmark.suite import (
    SuiteConfig,
    SuiteRunRecord,
    job_environment,
    load_suite_config,
    results_dir,
)
from unproject.benchmark.summary import SUMMARY_FILENAME, summarise
from unproject.benchmark.summary import to_csv as suite_to_csv
from unproject.bodies import device as device_module
from unproject.bodies.device import ENV_DEVICE, DeviceError, chip_name, requested_kind, resolve_kind
from unproject.config import ConfigError, load_config

_STARTED = datetime(2026, 9, 16, 9, 0, tzinfo=UTC)

_REGISTRY = """
schema_version = 1

[[machines]]
id = "A"
label = "a laptop"
device = "mps"

[[machines]]
id = "C-gpu"
label = "a rented box"
device = "cuda"
hardware_class = "rented-consumer-gpu"
"""


def _suite(**overrides: object) -> SuiteConfig:
    defaults: dict[str, object] = {
        "name": "suite",
        "warmup_frames": 1,
        "measured_frames": 2,
        "repetitions": 1,
        "cells": [{"name": "a-skeleton", "experiment": "fixture.toml", "output": "skeleton"}],
    }
    return SuiteConfig.model_validate(defaults | overrides)


class TestTheDeviceRequest:
    def test_unset_and_auto_mean_auto(self) -> None:
        assert requested_kind({}) == "auto"
        assert requested_kind({ENV_DEVICE: ""}) == "auto"
        assert requested_kind({ENV_DEVICE: " Auto "}) == "auto"

    def test_a_kind_is_read_case_insensitively(self) -> None:
        assert requested_kind({ENV_DEVICE: "CPU"}) == "cpu"
        assert requested_kind({ENV_DEVICE: "cuda"}) == "cuda"

    def test_a_typo_is_an_error_not_a_fallback(self) -> None:
        with pytest.raises(DeviceError, match="not a device"):
            requested_kind({ENV_DEVICE: "gpu"})


class TestResolvingTheDevice:
    def test_auto_takes_the_best_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(device_module, "available_kinds", lambda: ["cuda", "cpu"])
        assert resolve_kind("auto") == "cuda"
        monkeypatch.setattr(device_module, "available_kinds", lambda: ["mps", "cpu"])
        assert resolve_kind("auto") == "mps"
        monkeypatch.setattr(device_module, "available_kinds", lambda: ["cpu"])
        assert resolve_kind("auto") == "cpu"

    def test_a_named_kind_that_exists_is_taken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(device_module, "available_kinds", lambda: ["mps", "cpu"])
        assert resolve_kind("cpu") == "cpu"

    def test_a_named_kind_that_is_missing_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(device_module, "available_kinds", lambda: ["mps", "cpu"])
        with pytest.raises(DeviceError, match="only has mps, cpu"):
            resolve_kind("cuda")

    def test_the_environment_is_read_when_nothing_is_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(device_module, "available_kinds", lambda: ["mps", "cpu"])
        monkeypatch.setenv(ENV_DEVICE, "cpu")
        assert resolve_kind() == "cpu"


class TestTheSuiteDevicePolicy:
    def test_a_suite_may_name_a_device(self, write_config: WriteConfig) -> None:
        suite = load_suite_config(
            write_config(
                'name = "s"\ndevice = "cpu"\n[[cells]]\nname = "c"\nexperiment = "e.toml"\n',
                name="s.toml",
            )
        )
        assert suite.device == "cpu"

    def test_a_device_that_is_not_one_is_refused(self, write_config: WriteConfig) -> None:
        with pytest.raises(ConfigError, match="device"):
            load_suite_config(
                write_config(
                    'name = "s"\ndevice = "gpu"\n[[cells]]\nname = "c"\nexperiment = "e.toml"\n',
                    name="s.toml",
                )
            )

    def test_the_driver_hands_the_device_and_machine_to_every_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_DEVICE, raising=False)
        env = job_environment(_suite(device="cpu"), machine_id="B")
        assert env[ENV_DEVICE] == "cpu"
        assert env[ENV_MACHINE_ID] == "B"

    def test_an_auto_suite_leaves_the_environment_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_DEVICE, raising=False)
        env = job_environment(_suite(), machine_id=None)
        assert ENV_DEVICE not in env
        assert ENV_MACHINE_ID not in env

    def test_a_conflicting_variable_is_a_conflict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_DEVICE, "mps")
        with pytest.raises(ConfigError, match="asks for device 'cpu'"):
            job_environment(_suite(device="cpu"), machine_id=None)

    def test_a_job_holds_itself_to_the_suite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_DEVICE, raising=False)
        apply_device_policy(_suite(device="cpu"))
        assert device_module.requested_kind() == "cpu"

    def test_a_job_refuses_a_variable_that_disagrees(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_DEVICE, "cuda")
        with pytest.raises(BenchmarkError, match="asks for device 'cpu'"):
            apply_device_policy(_suite(device="cpu"))

    def test_the_run_record_carries_both(self) -> None:
        record = SuiteRunRecord(
            suite="s",
            suite_hash="a" * 64,
            order_seed=1,
            dry_run=False,
            started_at=_STARTED,
            python="python",
            command=["python"],
            machine_id="C-gpu",
            device="cuda",
        )
        again = SuiteRunRecord.model_validate_json(record.model_dump_json())
        assert (again.machine_id, again.device) == ("C-gpu", "cuda")

    def test_a_record_written_before_the_fields_still_reads(self) -> None:
        raw = {
            "suite": "s",
            "suite_hash": "a" * 64,
            "order_seed": 1,
            "dry_run": False,
            "started_at": _STARTED.isoformat(),
            "python": "python",
            "command": ["python"],
        }
        record = SuiteRunRecord.model_validate(raw)
        assert record.machine_id is None and record.device is None


class TestTheEnvironmentRecord:
    def test_a_result_written_before_the_fields_still_reads(self) -> None:
        old = {
            "machine": "Apple M3 Pro",
            "platform": "Darwin 25.0.0 arm64",
            "python_version": "3.12.7",
            "device": "mps",
            "package_versions": {},
        }
        environment = Environment.model_validate(old)
        assert environment.machine_id is None
        assert environment.device_name is None
        assert environment.torch_threads is None


class TestTheLinuxConditions:
    _LINE = "45, 28.50, 210, 405, 0x0000000000000004, P8, 3\n"

    def test_the_card_is_read_as_numbers(self) -> None:
        reading = parse_nvidia_smi(self._LINE)
        assert reading == AcceleratorReading(
            temperature_c=45.0,
            power_draw_w=28.5,
            sm_clock_mhz=210.0,
            memory_clock_mhz=405.0,
            throttle_reasons="0x0000000000000004",
            performance_state="P8",
            utilization_percent=3.0,
        )

    def test_a_field_the_driver_cannot_give_is_absent(self) -> None:
        reading = parse_nvidia_smi("45, [N/A], 210, 405, 0x0, P0, 0\n")
        assert reading is not None
        assert reading.power_draw_w is None
        assert reading.temperature_c == 45.0

    def test_the_first_card_is_the_one_recorded(self) -> None:
        reading = parse_nvidia_smi(self._LINE + "60, 200.0, 2500, 10000, 0x0, P0, 99\n")
        assert reading is not None
        assert reading.temperature_c == 45.0

    def test_a_line_of_the_wrong_shape_is_nothing(self) -> None:
        assert parse_nvidia_smi("") is None
        assert parse_nvidia_smi("45, 28.5\n") is None

    def test_conditions_carry_the_governor_the_card_and_the_load(self) -> None:
        conditions = parse_conditions(
            {"governor": "performance\n", "nvidia-smi": self._LINE}, load_average=0.42
        )
        assert conditions.cpu_governor == "performance"
        assert conditions.load_average_before == 0.42
        assert conditions.gpu_before is not None
        assert conditions.gpu_before.performance_state == "P8"
        assert conditions.power_source == "unknown"
        assert conditions.raw["nvidia-smi"] == self._LINE

    def test_the_mac_readings_are_unchanged(self) -> None:
        conditions = parse_conditions(
            {
                "batt": "Now drawing from 'AC Power'\n",
                "therm": "CPU_Speed_Limit \t= 100\n",
                "settings": " lowpowermode 0\n",
            },
            load_average=1.5,
        )
        assert conditions.power_source == "mains"
        assert conditions.cpu_speed_limit_before == 100
        assert conditions.low_power_mode is False
        assert conditions.gpu_before is None
        assert conditions.load_average_before == 1.5

    def test_the_readings_round_trip(self) -> None:
        conditions = parse_conditions({"nvidia-smi": self._LINE}, load_average=0.1)
        again = Conditions.model_validate_json(conditions.model_dump_json())
        assert again == conditions

    def test_a_card_busy_for_someone_else_stops_the_preflight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rented RTX 5090 (2026-09-15) read 100 %, P0, 98 W at rest with no
        process in this container's view: another tenant had the card."""
        import platform

        from unproject.benchmark import preflight

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(preflight, "_check_gpu_processes", lambda: iter(()))

        def verdict(line: str) -> str:
            conditions = parse_conditions({"nvidia-smi": line}, load_average=0.1)
            checks = list(preflight._check_conditions(conditions, "cuda"))
            return next(check.status for check in checks if check.name == "gpu")

        assert verdict(self._LINE) == "ok"
        assert verdict("45, 28.50, 210, 405, 0x0000000000000000, P2, 12\n") == "warn"
        assert verdict("61, 98.05, 2850, 405, 0x0000000000000000, P0, 100\n") == "fail"


def _on(machine_id: str | None, device: str = "mps", **overrides: object) -> Environment:
    return Environment(
        machine="Apple M4",
        platform="Darwin 25.6.0 arm64",
        python_version="3.12.14",
        device=device,
        package_versions={},
        git_commit="c" * 40,
        machine_id=machine_id,
        device_name="Apple M4" if device == "mps" else "NVIDIA GeForce RTX 4070",
    )


class TestTheSummaryKnowsTheMachine:
    def test_one_machine_on_one_device_meets_the_protocol(self) -> None:
        results = [
            _result([0.5] * 60, repetition=r, job_index=r, environment=_on("B")) for r in range(3)
        ]
        summary = summarise("suite", results)
        cell = summary.cells[0]
        assert cell.machine_ids == ["B"]
        assert cell.devices == ["mps"]
        assert cell.device_names == ["Apple M4"]
        assert cell.one_machine is True
        assert cell.meets_protocol is True

    def test_two_machines_in_one_cell_do_not(self) -> None:
        results = [
            _result([0.5] * 60, repetition=0, job_index=0, environment=_on("A")),
            _result([0.5] * 60, repetition=1, job_index=1, environment=_on("B")),
            _result([0.5] * 60, repetition=2, job_index=2, environment=_on("B")),
        ]
        cell = summarise("suite", results).cells[0]
        assert cell.machine_ids == ["A", "B"]
        assert cell.one_machine is False
        assert cell.meets_protocol is False

    def test_two_devices_on_one_machine_do_not_either(self) -> None:
        results = [
            _result([0.5] * 60, repetition=0, job_index=0, environment=_on("B", "cpu")),
            _result([0.5] * 60, repetition=1, job_index=1, environment=_on("B", "mps")),
            _result([0.5] * 60, repetition=2, job_index=2, environment=_on("B", "mps")),
        ]
        cell = summarise("suite", results).cells[0]
        assert cell.devices == ["cpu", "mps"]
        assert cell.meets_protocol is False

    def test_an_undeclared_machine_is_named_as_such(self) -> None:
        cell = summarise("suite", [_result([0.5] * 4, environment=_on(None))]).cells[0]
        assert cell.machine_ids == ["undeclared"]

    def test_the_median_per_job_is_kept_beside_the_pooled_one(self) -> None:
        results = [
            _result([0.4, 0.4, 0.4, 0.4], repetition=0, job_index=0),
            _result([0.6, 0.6, 0.6, 0.6], repetition=1, job_index=1),
        ]
        cell = summarise("suite", results).cells[0]
        assert cell.inference_median_per_job == [0.4, 0.6]
        assert cell.inference.median == 0.5

    def test_the_load_average_reaches_the_summary(self) -> None:
        results = [
            _result(
                [0.5] * 4,
                conditions=Conditions(load_average_before=1.0, load_average_after=2.5),
            )
        ]
        cell = summarise("suite", results).cells[0]
        assert cell.load_average_max == 2.5

    def test_the_csv_names_the_machine_and_device(self) -> None:
        summary = summarise("suite", [_result([0.5] * 4, environment=_on("B"))])
        header, row = suite_to_csv(summary).splitlines()[:2]
        columns = header.split(",")
        values = row.split(",")
        assert values[columns.index("machine_id")] == "B"
        assert values[columns.index("device")] == "mps"
        assert values[columns.index("inference_median_per_job_s")] == "0.500000"
        assert values[columns.index("git_commit")] == "c" * 12


class TestTheRegistry:
    def test_it_reads_the_machines(self, write_config: WriteConfig) -> None:
        registry = load_registry(write_config(_REGISTRY, name="machines.toml"))
        assert [m.id for m in registry.machines] == ["A", "C-gpu"]
        found = registry.find("C-gpu")
        assert found is not None and found.device == "cuda"
        assert found.hardware_class == "rented-consumer-gpu"
        assert registry.find("nope") is None

    def test_an_id_declared_twice_is_refused(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            MachineRegistry.model_validate(
                {
                    "machines": [
                        {"id": "A", "label": "x", "device": "mps"},
                        {"id": "A", "label": "y", "device": "cpu"},
                    ]
                }
            )

    def test_an_id_is_a_directory_name(self) -> None:
        with pytest.raises(ValueError, match="pattern"):
            MachineRegistry.model_validate(
                {"machines": [{"id": "a/b", "label": "x", "device": "mps"}]}
            )

    def test_the_real_registry_reads(self) -> None:
        path = Path(__file__).resolve().parents[3] / "benchmarks" / "machines.toml"
        registry = load_registry(path)
        assert {m.id for m in registry.machines} >= {"A", "B"}


def _write_suite_summary(root: Path, machine_id: str, suite: str, recorded_as: str) -> Path:
    results = [
        _result([0.5] * 60, repetition=r, job_index=r, environment=_on(recorded_as))
        for r in range(3)
    ]
    summary = summarise(suite, results)
    directory = results_dir(machine_root(root, machine_id), suite)
    directory.mkdir(parents=True)
    path = directory / SUMMARY_FILENAME
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return path


class TestTheCampaign:
    def test_it_folds_one_summary_per_machine_and_suite(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        _write_suite_summary(root, "A", "matrix", recorded_as="A")
        _write_suite_summary(root, "A", "people", recorded_as="A")
        _write_suite_summary(root, "B", "matrix", recorded_as="B")

        campaign = read_campaign(root)

        assert campaign.machines == ["A", "B"]
        assert campaign.suites == ["matrix", "people"]
        assert [(c.machine_id, c.suite) for c in campaign.cells] == [
            ("A", "matrix"),
            ("A", "people"),
            ("B", "matrix"),
        ]
        assert all(c.consistent for c in campaign.cells)
        assert campaign.missing == []

    def test_a_result_that_names_another_machine_is_flagged(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        _write_suite_summary(root, "B", "matrix", recorded_as="A")

        campaign = read_campaign(root)

        assert campaign.cells[0].consistent is False
        assert campaign.cells[0].recorded_machine_ids == ["A"]

    def test_a_machine_without_results_is_named(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        _write_suite_summary(root, "A", "matrix", recorded_as="A")
        (root / "C-gpu").mkdir()
        (root / "B" / "benchmark-v2" / "matrix").mkdir(parents=True)

        campaign = read_campaign(root)

        assert campaign.machines == ["A"]
        assert "C-gpu: no results" in campaign.missing
        assert any(item.startswith("B/matrix: no summary") for item in campaign.missing)

    def test_the_registry_is_checked_not_enforced(
        self, tmp_path: Path, write_config: WriteConfig
    ) -> None:
        root = tmp_path / "campaign"
        _write_suite_summary(root, "Z", "matrix", recorded_as="Z")
        registry = load_registry(write_config(_REGISTRY, name="machines.toml"))

        campaign = read_campaign(root, registry=registry)

        assert campaign.machines == ["Z"]
        assert "Z: not in the registry" in campaign.missing

    def test_it_writes_json_and_csv_beside_the_machines(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        _write_suite_summary(root, "A", "matrix", recorded_as="A")
        campaign = read_campaign(root)

        as_json, as_csv, as_pairs = write_campaign(campaign, root)

        assert as_json == root / CAMPAIGN_FILENAME
        assert as_csv == root / CAMPAIGN_CSV_FILENAME
        assert json.loads(as_json.read_text("utf-8"))["machines"] == ["A"]
        header, row = to_csv(campaign).splitlines()
        columns = header.split(",")
        values = row.split(",")
        assert values[columns.index("machine_id")] == "A"
        assert values[columns.index("inference_median_s")] == "0.5"
        assert values[columns.index("consistent")] == "true"
        assert as_pairs.read_text("utf-8").startswith("machine_id,suite,a,b")

    def test_no_campaign_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="no campaign"):
            read_campaign(tmp_path / "nowhere")


_EXPERIMENT = """
name = "fixture"
[ingest]
source_path = "clip.avi"
"""

_EXPERIMENT_WITH_DIGEST = (
    _EXPERIMENT
    + """
[provenance]
title = "t"
author = "a"
source_url = "u"
licence = "l"
acquired_on = 2026-09-15
original_sha256 = "%s"
original_bytes = 1
derivation = "d"
expected_sha256 = "%s"
"""
)


@pytest.fixture
def quiet_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean tree on a declared machine with the estimator installed, whatever
    the test host is: the checks under test are the ones about the inputs."""
    import importlib.util

    from unproject.run import manifest

    monkeypatch.setattr(manifest, "git_commit", lambda: "c" * 40)
    monkeypatch.setattr(manifest, "git_dirty", lambda: False)
    # Only the estimator is pretended; every other lookup stays real, because
    # torch's own import asks `find_spec` about optional modules and caches
    # the answer for the life of the process.
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None) -> object:
        return object() if name == "multihmr2" else real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(device_module, "available_kinds", lambda: ["cpu"])
    monkeypatch.delenv(ENV_DEVICE, raising=False)


def _report(
    tmp_path: Path,
    write_config: WriteConfig,
    experiment: str,
    *,
    machine_id: str | None = "A",
    registry: MachineRegistry | None = None,
    suite: Callable[[], SuiteConfig] = lambda: _suite(device="cpu"),
) -> PreflightReport:
    write_config(experiment, name="fixture.toml")
    return run_preflight(
        suite(),
        suite_dir=tmp_path,
        artifacts_root=tmp_path / "artifacts",
        machine_id=machine_id,
        registry=registry,
    )


def _checks(report: PreflightReport, name: str) -> list[Check]:
    return [check for check in report.checks if check.name == name]


@pytest.mark.usefixtures("quiet_machine")
class TestThePreflight:
    def test_a_clean_declared_machine_with_its_inputs_is_ready(
        self, tmp_path: Path, make_video: MakeVideo, write_config: WriteConfig
    ) -> None:
        make_video(frame_count=6)
        report = _report(tmp_path, write_config, _EXPERIMENT)

        assert _checks(report, "git")[0].status == "ok"
        assert _checks(report, "device")[0].status == "ok"
        assert _checks(report, "input:a-skeleton")[0].status == "warn"  # no digest declared
        assert report.ok is True

    def test_a_missing_input_is_a_failure(self, tmp_path: Path, write_config: WriteConfig) -> None:
        report = _report(tmp_path, write_config, _EXPERIMENT)

        check = _checks(report, "input:a-skeleton")[0]
        assert check.status == "fail"
        assert "does not exist" in check.detail
        assert report.ok is False

    def test_the_campaigns_bytes_are_checked(
        self, tmp_path: Path, make_video: MakeVideo, write_config: WriteConfig
    ) -> None:
        from unproject.ingest.hashing import file_sha256

        path = make_video(frame_count=6)
        digest = file_sha256(path)
        report = _report(tmp_path, write_config, _EXPERIMENT_WITH_DIGEST % (digest, digest))
        assert _checks(report, "input:a-skeleton")[0].status == "ok"

        other = "0" * 64
        report = _report(tmp_path, write_config, _EXPERIMENT_WITH_DIGEST % (digest, other))
        check = _checks(report, "input:a-skeleton")[0]
        assert check.status == "fail"
        assert "not the same bytes" in check.detail

    def test_a_clip_too_short_for_the_policy_is_a_failure(
        self, tmp_path: Path, make_video: MakeVideo, write_config: WriteConfig
    ) -> None:
        make_video(frame_count=2)
        report = _report(tmp_path, write_config, _EXPERIMENT)
        check = _checks(report, "input:a-skeleton")[0]
        assert check.status == "fail"
        assert "the suite needs 3" in check.detail

    def test_an_undeclared_machine_is_a_failure(
        self, tmp_path: Path, make_video: MakeVideo, write_config: WriteConfig
    ) -> None:
        make_video(frame_count=6)
        registry = MachineRegistry.model_validate(
            {"machines": [{"id": "A", "label": "x", "device": "cpu"}]}
        )
        report = _report(tmp_path, write_config, _EXPERIMENT, machine_id="Q", registry=registry)
        check = _checks(report, "machine")[0]
        assert check.status == "fail"
        assert "not declared" in check.detail

        report = _report(tmp_path, write_config, _EXPERIMENT, machine_id=None)
        assert _checks(report, "machine")[0].status == "fail"

    def test_the_registry_says_what_device_to_expect(
        self, tmp_path: Path, make_video: MakeVideo, write_config: WriteConfig
    ) -> None:
        make_video(frame_count=6)
        registry = MachineRegistry.model_validate(
            {"machines": [{"id": "A", "label": "x", "device": "cuda"}]}
        )
        # An auto suite on a machine declared as CUDA that only has a CPU.
        report = _report(
            tmp_path, write_config, _EXPERIMENT, registry=registry, suite=lambda: _suite()
        )
        check = _checks(report, "device")[0]
        assert check.status == "fail"
        assert "expects cuda" in check.detail

        # A CPU suite on the same machine is what it says it is.
        report = _report(tmp_path, write_config, _EXPERIMENT, registry=registry)
        assert _checks(report, "device")[0].status == "ok"

    def test_a_dirty_tree_is_a_failure(
        self,
        tmp_path: Path,
        make_video: MakeVideo,
        write_config: WriteConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from unproject.run import manifest

        monkeypatch.setattr(manifest, "git_dirty", lambda: True)
        make_video(frame_count=6)
        report = _report(tmp_path, write_config, _EXPERIMENT)
        assert _checks(report, "git")[0].status == "fail"
        assert report.ok is False

    def test_existing_results_are_a_warning(
        self, tmp_path: Path, make_video: MakeVideo, write_config: WriteConfig
    ) -> None:
        make_video(frame_count=6)
        directory = results_dir(tmp_path / "artifacts", "suite") / "a-skeleton"
        directory.mkdir(parents=True)
        (directory / "old.json").write_text("{}", encoding="utf-8")
        report = _report(tmp_path, write_config, _EXPERIMENT)
        check = _checks(report, "results")[0]
        assert check.status == "warn"
        assert "already holds 1 file" in check.detail

    def test_the_report_is_written_beside_the_results_and_nothing_else_is(
        self, tmp_path: Path, make_video: MakeVideo, write_config: WriteConfig
    ) -> None:
        make_video(frame_count=6)
        write_config(_EXPERIMENT, name="fixture.toml")
        before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
        report = _report(tmp_path, write_config, _EXPERIMENT)
        path = write_report(report, tmp_path / "artifacts")

        assert path.name.startswith(PREFLIGHT_PREFIX)
        assert path.parent == results_dir(tmp_path / "artifacts", "suite")
        again = PreflightReport.model_validate_json(path.read_text("utf-8"))
        assert again.ok == report.ok
        after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
        new = set(after) - set(before)
        assert all(str(item).startswith("artifacts") for item in new)


class TestTheExperimentProvenance:
    def test_the_archive_and_deposit_fields_are_optional(self, write_config: WriteConfig) -> None:
        config = load_config(
            write_config(_EXPERIMENT_WITH_DIGEST % ("a" * 64, "b" * 64), name="e.toml")
        )
        assert config.provenance is not None
        assert config.provenance.expected_sha256 == "b" * 64
        assert config.provenance.archived_url is None
        assert config.provenance.deposit_doi is None

    def test_an_expected_digest_is_a_digest(self, write_config: WriteConfig) -> None:
        with pytest.raises(ConfigError, match="expected_sha256"):
            load_config(
                write_config(_EXPERIMENT_WITH_DIGEST % ("a" * 64, "not-a-digest"), name="e.toml")
            )

    def test_the_committed_experiments_pin_the_campaigns_bytes(self) -> None:
        root = Path(__file__).resolve().parents[3] / "experiments"
        for name in (
            "mot16-09",
            "mot16-09-half",
            "mot16-09-letterbox",
            "mot16-09-letterbox-half",
            "mot16-08",
            "mupots-ts10",
            "mupots-ts11",
        ):
            config = load_config(root / f"{name}.toml")
            assert config.provenance is not None, name
            assert config.provenance.expected_sha256 is not None, name


def test_the_started_stamp_orders_reports(tmp_path: Path) -> None:
    first = _STARTED
    second = _STARTED + timedelta(seconds=1)
    assert first.strftime("%Y%m%dT%H%M%S%f") < second.strftime("%Y%m%dT%H%M%S%f")


@pytest.mark.usefixtures("quiet_machine")
def test_the_preflight_describes_the_device_the_suite_will_use(
    tmp_path: Path,
    make_video: MakeVideo,
    write_config: WriteConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found on the CUDA shakedown (2026-09-15): the report described the
    device `auto` would pick, so a CPU suite on a host with a card read
    `cpu: <the card>` and recorded `device: cuda` beside `device_requested:
    cpu`. Here the machine offers an accelerator and the suite asks for the
    CPU; the report must say cpu."""
    pytest.importorskip("torch")
    monkeypatch.setattr(device_module, "available_kinds", lambda: ["mps", "cpu"])
    make_video(frame_count=6)

    report = _report(tmp_path, write_config, _EXPERIMENT)

    assert report.device_requested == "cpu"
    assert report.environment.device == "cpu"
    assert report.environment.cuda_version is None
    assert report.environment.device_name == chip_name()
    assert _checks(report, "device")[0].detail.startswith(f"cpu: {chip_name()}")
