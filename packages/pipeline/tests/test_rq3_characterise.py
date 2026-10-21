"""Tests for the filter characterisation and the second shared vector.
ADR-0005 WP6.

The metrics are checked on signals whose answer is known by construction: a
filter that does nothing, a step that is never reached, a sine at a frequency
the filter passes. The parity fixture is checked against the Python filter
here and against the viewer's in `bodies.test.ts`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from unproject.rq3.characterise import (
    CADENCES_HZ,
    DISTANCES_M,
    REPORT_FILENAME,
    SEED,
    Characterisation,
    Signal,
    characterise,
    measure,
    parity_fixture,
    run_filter,
    signals,
    write_characterisation,
)
from unproject.rq3.smoothing import DEFAULT_SMOOTHING, TrackSmoother

_FIXTURES = Path(__file__).parent / "fixtures"


class TestTheSuite:
    def test_it_covers_the_viewers_cadences_and_distances(self) -> None:
        seen = {(s.axis, s.cadence_hz, s.distance_m) for s in signals()}

        for cadence in CADENCES_HZ:
            for distance in DISTANCES_M:
                assert ("depth", cadence, distance) in seen
        assert ("lateral", 25.0, 8.0) in seen

    def test_no_sine_is_above_nyquist(self) -> None:
        for signal in signals():
            if signal.kind == "sine":
                assert signal.frequency_hz is not None
                assert signal.frequency_hz < signal.cadence_hz / 2.0

    def test_it_is_deterministic(self) -> None:
        first = characterise(seed=SEED).model_dump_json()
        second = characterise(seed=SEED).model_dump_json()

        assert first == second
        assert characterise(seed=SEED + 1).model_dump_json() != first

    def test_it_writes_its_files(self, tmp_path: Path) -> None:
        written = write_characterisation(characterise(), tmp_path)

        assert [p.name for p in written] == [
            REPORT_FILENAME,
            "filter-characterisation.csv",
            "filter-attenuation.svg",
            "filter-step-response.svg",
        ]
        report = Characterisation.model_validate_json((tmp_path / REPORT_FILENAME).read_text())
        assert report.options == DEFAULT_SMOOTHING
        rows = (tmp_path / "filter-characterisation.csv").read_text().splitlines()
        assert len(rows) == len(report.results) + 1


def _signal(
    kind: str, times: np.ndarray, clean: np.ndarray, raw: np.ndarray, **extra: object
) -> Signal:
    return Signal(
        kind=kind,  # type: ignore[arg-type]
        axis="depth",
        cadence_hz=10.0,
        distance_m=8.0,
        frequency_hz=extra.get("frequency_hz"),  # type: ignore[arg-type]
        times=times,
        clean=clean,
        raw=raw,
    )


class TestTheMetrics:
    def test_a_filter_that_passes_the_noise_through_attenuates_nothing(self) -> None:
        rng = np.random.default_rng(0)
        times = np.arange(100) / 10.0
        clean = np.full(100, 8.0)
        raw = clean + rng.normal(0.0, 0.2, 100)

        result = measure(_signal("noise", times, clean, raw), raw.copy())

        assert result.attenuation_ratio == pytest.approx(1.0)
        assert result.rms_raw_m == result.rms_filtered_m

    def test_a_perfect_filter_attenuates_everything(self) -> None:
        rng = np.random.default_rng(0)
        times = np.arange(100) / 10.0
        clean = np.full(100, 8.0)

        result = measure(_signal("noise", times, clean, clean + rng.normal(0, 0.2, 100)), clean)

        assert result.rms_filtered_m == 0.0
        assert result.attenuation_ratio == 0.0

    def test_a_step_reached_immediately_has_no_delay_and_no_overshoot(self) -> None:
        times = np.arange(100) / 10.0
        clean = np.where(times >= 2.0, 9.0, 8.0)

        result = measure(_signal("step", times, clean, clean), clean.copy())

        assert result.t90_seconds == 0.0
        assert result.settling_seconds == 0.0
        assert result.overshoot_fraction == 0.0

    def test_a_step_never_reached_has_no_t90_and_no_settling(self) -> None:
        times = np.arange(100) / 10.0
        clean = np.where(times >= 2.0, 9.0, 8.0)

        result = measure(_signal("step", times, clean, clean), np.full(100, 8.0))

        assert result.t90_seconds is None
        assert result.settling_seconds is None

    def test_overshoot_is_measured_as_a_fraction_of_the_step(self) -> None:
        times = np.arange(100) / 10.0
        clean = np.where(times >= 2.0, 9.0, 8.0)
        filtered = clean.copy()
        filtered[30] = 9.25

        result = measure(_signal("step", times, clean, clean), filtered)

        assert result.overshoot_fraction == pytest.approx(0.25)

    def test_a_ramp_followed_with_a_constant_lag_reports_it(self) -> None:
        times = np.arange(80) / 10.0
        clean = 8.0 + 1.0 * times

        result = measure(_signal("ramp", times, clean, clean), clean - 0.4)

        assert result.lag_metres == pytest.approx(0.4)
        assert result.lag_seconds == pytest.approx(0.4)

    def test_a_sine_passed_unchanged_has_zero_gain_and_no_delay(self) -> None:
        times = np.arange(400) / 10.0
        clean = 8.0 + 0.5 * np.sin(2 * math.pi * 0.5 * times)

        result = measure(_signal("sine", times, clean, clean, frequency_hz=0.5), clean.copy())

        assert result.gain_db == pytest.approx(0.0, abs=1e-6)
        assert result.phase_delay_seconds == pytest.approx(0.0, abs=1e-6)

    def test_a_halved_delayed_sine_reads_minus_6_db_and_the_delay(self) -> None:
        times = np.arange(400) / 10.0
        clean = 8.0 + 0.5 * np.sin(2 * math.pi * 0.5 * times)
        filtered = 8.0 + 0.25 * np.sin(2 * math.pi * 0.5 * (times - 0.3))

        result = measure(_signal("sine", times, clean, clean, frequency_hz=0.5), filtered)

        assert result.gain_db == pytest.approx(20 * math.log10(0.5), abs=1e-3)
        assert result.phase_delay_seconds == pytest.approx(0.3, abs=1e-3)


class TestWhatTheFilterDoes:
    """Properties the README claims, now as numbers on a known signal."""

    def test_depth_noise_is_attenuated_more_at_range(self) -> None:
        report = characterise()
        by_distance = {
            r.distance_m: _number(r.attenuation_ratio)
            for r in report.results
            if r.kind == "noise" and r.axis == "depth" and r.cadence_hz == 10.0
        }

        assert by_distance[40.0] < by_distance[8.0] < by_distance[2.0]
        assert all(ratio < 1.0 for ratio in by_distance.values())

    def test_the_price_of_that_is_delay_that_grows_with_range(self) -> None:
        report = characterise()
        t90 = {
            r.distance_m: _number(r.t90_seconds)
            for r in report.results
            if r.kind == "step" and r.axis == "depth" and r.cadence_hz == 10.0
        }
        lag = {
            r.distance_m: _number(r.lag_seconds)
            for r in report.results
            if r.kind == "ramp" and r.axis == "depth" and r.cadence_hz == 10.0
        }

        assert t90[2.0] < t90[8.0] < t90[40.0]
        assert lag[2.0] < lag[8.0] < lag[40.0]

    def test_higher_frequencies_are_attenuated_more(self) -> None:
        report = characterise()
        gains = sorted(
            (_number(r.frequency_hz), _number(r.gain_db))
            for r in report.results
            if r.kind == "sine"
            and r.axis == "depth"
            and r.cadence_hz == 25.0
            and r.distance_m == 8.0
        )

        assert [g for _, g in gains] == sorted((g for _, g in gains), reverse=True)

    def test_the_depth_filter_never_overshoots(self) -> None:
        report = characterise()

        assert all(
            r.overshoot_fraction == 0.0
            for r in report.results
            if r.kind == "step" and r.axis == "depth"
        )


class TestTheSharedSignalVector:
    """`one_euro_signals.json` is read by both suites. If either implementation
    drifts, both fail."""

    def test_the_committed_fixture_is_what_this_code_generates(self) -> None:
        committed = json.loads((_FIXTURES / "one_euro_signals.json").read_text("utf-8"))

        assert committed == parity_fixture()

    def test_the_python_filter_reproduces_the_fixture(self) -> None:
        fixture = json.loads((_FIXTURES / "one_euro_signals.json").read_text("utf-8"))

        for case in fixture["cases"]:
            smoother = TrackSmoother(DEFAULT_SMOOTHING)
            for sample, want in zip(case["samples"], case["expected"], strict=True):
                [(_, root)] = smoother.frame(
                    sample["time"], [(0, (sample["x"], sample["y"], sample["z"]))]
                )
                assert list(root) == pytest.approx(want, abs=1e-6)

    def test_the_fixture_spans_the_cadences_and_distances_it_claims(self) -> None:
        fixture = json.loads((_FIXTURES / "one_euro_signals.json").read_text("utf-8"))

        assert {(c["cadence_hz"], c["distance_m"]) for c in fixture["cases"]} == {
            (2.5, 2.0),
            (2.5, 40.0),
            (25.0, 2.0),
            (25.0, 40.0),
        }
        assert fixture["options"]["depth"]["cutoffHz"] == DEFAULT_SMOOTHING.depth.cutoff_hz


def _number(value: float | None) -> float:
    assert value is not None
    return value


def test_run_filter_uses_the_axis_it_was_told() -> None:
    times = np.arange(20) / 10.0
    clean = np.full(20, 0.4)
    lateral = Signal("noise", "lateral", 10.0, 8.0, None, times, clean, clean)
    depth = Signal("noise", "depth", 10.0, 8.0, None, times, np.full(20, 8.0), np.full(20, 8.0))

    assert run_filter(lateral, DEFAULT_SMOOTHING)[-1] == pytest.approx(0.4)
    assert run_filter(depth, DEFAULT_SMOOTHING)[-1] == pytest.approx(8.0)
