"""What the root filter does to a signal whose truth is known.

ADR-0005 WP6. On real footage the filter is scored for *smoothness* - the
apparent depth velocity it removes - and never for accuracy, because the
footage has no truth to be accurate against. This module gives it one: a set
of synthetic root trajectories, each with a known clean value under known
noise, run through the same `TrackSmoother` the analysis uses, and measured
for what a filter is measured for - noise attenuation, settling time,
overshoot, tracking delay, and attenuation per frequency.

The suite is deterministic: signals are built from a fixed seed, and the
report regenerates byte for byte. It covers the sample cadences the viewer
sees - a camera answered at about 2.5 fps, a clip at stride 2 at about 10 fps,
a 25 fps clip at stride 1 - and the distances the range-proportional depth
filter is designed around, from arm's length to the far end of a crowd.

The viewer's own implementation, `apps/web/src/bodies.ts`, is held to this one
by `tests/fixtures/one_euro_signals.json`, a subset of these signals with the
Python outputs recorded; the TypeScript test replays it and both suites fail if
either drifts.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from unproject.rq3.smoothing import DEFAULT_SMOOTHING, Smoothing, TrackSmoother

if TYPE_CHECKING:
    from matplotlib.figure import Figure

CHARACTERISATION_SCHEMA_VERSION = 1
REPORT_FILENAME = "filter-characterisation.json"
TABLE_FILENAME = "filter-characterisation.csv"
ATTENUATION_FIGURE = "filter-attenuation.svg"
STEP_FIGURE = "filter-step-response.svg"

# The cadences the viewer actually sees, in samples per second.
CADENCES_HZ = (2.5, 10.0, 25.0)
# Distances spanning what the crowd clips contain and what the depth filter's
# range term is clamped over: 8 m is the reference, 40 m the far end.
DISTANCES_M = (2.0, 8.0, 20.0, 40.0)
# Sinusoid frequencies; only those below the Nyquist limit of a cadence run.
FREQUENCIES_HZ = (0.1, 0.25, 0.5, 1.0, 2.0)
# The measured depth jitter is about 3 % of range (README); the noise signal
# uses that so the attenuation is of the noise the filter was tuned for.
NOISE_FRACTION_OF_RANGE = 0.03
NOISE_SAMPLES = 400
STEP_METRES = 1.0
STEP_AT_SECONDS = 2.0
STEP_SECONDS = 10.0
RAMP_METRES_PER_SECOND = 1.0
RAMP_SECONDS = 8.0
SINE_AMPLITUDE_M = 0.5
SINE_PERIODS = 4.0
SEED = 20260915

# Settling: the step response has settled when it stays within this fraction
# of the step of its final value.
SETTLE_BAND = 0.1

type SignalKind = Literal["noise", "step", "ramp", "sine"]
type Axis = Literal["depth", "lateral"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SignalResult(_Frozen):
    """One synthetic signal through the filter, and what came of it."""

    kind: SignalKind
    axis: Axis
    cadence_hz: float
    distance_m: float
    frequency_hz: float | None = None
    samples: int
    # Noise: root-mean-square distance from the clean value, before and after.
    rms_raw_m: float | None = None
    rms_filtered_m: float | None = None
    attenuation_ratio: float | None = None
    # Step: seconds from the step to the first sample within 90 % of it, to
    # the first sample after which every sample stays within 10 % of it, and
    # the overshoot as a fraction of the step.
    t90_seconds: float | None = None
    settling_seconds: float | None = None
    overshoot_fraction: float | None = None
    # Ramp: how far behind the filtered value runs once it has caught up to
    # the slope, as a delay in seconds and a distance in metres.
    lag_seconds: float | None = None
    lag_metres: float | None = None
    # Sine: output over input amplitude, in dB, and the phase delay.
    gain_db: float | None = None
    phase_delay_seconds: float | None = None


class Characterisation(_Frozen):
    schema_version: int = Field(default=CHARACTERISATION_SCHEMA_VERSION, ge=1)
    options: Smoothing
    seed: int
    cadences_hz: list[float]
    distances_m: list[float]
    frequencies_hz: list[float]
    results: list[SignalResult]
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Signal:
    kind: SignalKind
    axis: Axis
    cadence_hz: float
    distance_m: float
    frequency_hz: float | None
    times: np.ndarray
    clean: np.ndarray
    raw: np.ndarray


def signals(seed: int = SEED) -> Iterator[Signal]:
    """Every synthetic signal of the suite, in a fixed order."""
    rng = np.random.default_rng(seed)
    for axis in ("depth", "lateral"):
        for cadence in CADENCES_HZ:
            for distance in DISTANCES_M:
                if axis == "lateral" and distance != DISTANCES_M[1]:
                    # The lateral filter does not follow range; one distance
                    # says everything, at the reference.
                    continue
                dt = 1.0 / cadence
                # Constant with noise.
                times = np.arange(NOISE_SAMPLES) * dt
                clean = np.full(NOISE_SAMPLES, distance)
                sigma = NOISE_FRACTION_OF_RANGE * distance
                yield Signal(
                    "noise",
                    axis,
                    cadence,
                    distance,
                    None,
                    times,
                    clean,
                    clean + rng.normal(0.0, sigma, NOISE_SAMPLES),
                )
                # Step.
                count = round(STEP_SECONDS * cadence)
                times = np.arange(count) * dt
                clean = np.where(times >= STEP_AT_SECONDS, distance + STEP_METRES, distance)
                yield Signal("step", axis, cadence, distance, None, times, clean, clean.copy())
                # Ramp.
                count = round(RAMP_SECONDS * cadence)
                times = np.arange(count) * dt
                clean = distance + RAMP_METRES_PER_SECOND * times
                yield Signal("ramp", axis, cadence, distance, None, times, clean, clean.copy())
                # Sinusoids below Nyquist.
                for frequency in FREQUENCIES_HZ:
                    if frequency >= cadence / 2.0:
                        continue
                    count = round(SINE_PERIODS / frequency * cadence)
                    times = np.arange(count) * dt
                    clean = distance + SINE_AMPLITUDE_M * np.sin(2.0 * math.pi * frequency * times)
                    yield Signal(
                        "sine", axis, cadence, distance, frequency, times, clean, clean.copy()
                    )


def run_filter(signal: Signal, options: Smoothing) -> np.ndarray:
    """The signal through the viewer's filter, on the axis it belongs to."""
    smoother = TrackSmoother(options)
    out = np.empty_like(signal.raw)
    for index, (time, value) in enumerate(zip(signal.times, signal.raw, strict=True)):
        if signal.axis == "depth":
            root = (0.0, 0.0, float(value))
        else:
            # Lateral at the reference range, so the depth term is neutral.
            root = (float(value), 0.0, DISTANCES_M[1])
        [(_, filtered)] = smoother.frame(float(time), [(0, root)])
        out[index] = filtered[2] if signal.axis == "depth" else filtered[0]
    return out


def measure(signal: Signal, filtered: np.ndarray) -> SignalResult:
    base = {
        "kind": signal.kind,
        "axis": signal.axis,
        "cadence_hz": signal.cadence_hz,
        "distance_m": signal.distance_m,
        "frequency_hz": signal.frequency_hz,
        "samples": int(signal.times.shape[0]),
    }
    if signal.kind == "noise":
        rms_raw = float(np.sqrt(np.mean((signal.raw - signal.clean) ** 2)))
        rms_filtered = float(np.sqrt(np.mean((filtered - signal.clean) ** 2)))
        return SignalResult(
            **base,
            rms_raw_m=_round(rms_raw),
            rms_filtered_m=_round(rms_filtered),
            attenuation_ratio=_round(rms_filtered / rms_raw) if rms_raw > 0.0 else None,
        )
    if signal.kind == "step":
        return SignalResult(**base, **_step_metrics(signal, filtered))
    if signal.kind == "ramp":
        lag_metres = float(np.mean((signal.clean - filtered)[-int(signal.cadence_hz) :]))
        return SignalResult(
            **base,
            lag_metres=_round(lag_metres),
            lag_seconds=_round(lag_metres / RAMP_METRES_PER_SECOND),
        )
    return SignalResult(**base, **_sine_metrics(signal, filtered))


def _step_metrics(signal: Signal, filtered: np.ndarray) -> dict[str, float | None]:
    after = signal.times >= STEP_AT_SECONDS
    response = filtered[after] - signal.distance_m
    times = signal.times[after] - STEP_AT_SECONDS
    reached = np.nonzero(response >= 0.9 * STEP_METRES)[0]
    t90 = float(times[reached[0]]) if reached.size else None
    # Settled: the first sample after which every sample stays in the band.
    outside = np.nonzero(np.abs(response - STEP_METRES) > SETTLE_BAND * STEP_METRES)[0]
    settling: float | None = 0.0
    if outside.size:
        settling = float(times[outside[-1] + 1]) if outside[-1] + 1 < times.shape[0] else None
    overshoot = float(max(0.0, response.max() - STEP_METRES) / STEP_METRES)
    return {
        "t90_seconds": _round(t90) if t90 is not None else None,
        "settling_seconds": _round(settling) if settling is not None else None,
        "overshoot_fraction": _round(overshoot),
    }


def _sine_metrics(signal: Signal, filtered: np.ndarray) -> dict[str, float | None]:
    """Fit the output to a sine and cosine at the known frequency, after the
    first period so the start-up transient is out of the fit."""
    assert signal.frequency_hz is not None
    omega = 2.0 * math.pi * signal.frequency_hz
    settled = signal.times >= 1.0 / signal.frequency_hz
    times = signal.times[settled]
    centred = filtered[settled] - signal.distance_m
    design = np.stack([np.sin(omega * times), np.cos(omega * times), np.ones_like(times)], axis=1)
    (a, b, _), *_ = np.linalg.lstsq(design, centred, rcond=None)
    amplitude = float(math.hypot(a, b))
    # a·sin + b·cos = A·sin(ωt + φ) with φ = atan2(b, a); a lag is φ < 0.
    phase = math.atan2(b, a)
    delay = -phase / omega
    gain = 20.0 * math.log10(amplitude / SINE_AMPLITUDE_M) if amplitude > 0.0 else -math.inf
    return {
        "gain_db": _round(gain) if math.isfinite(gain) else None,
        "phase_delay_seconds": _round(delay),
    }


def characterise(options: Smoothing = DEFAULT_SMOOTHING, *, seed: int = SEED) -> Characterisation:
    results = [measure(signal, run_filter(signal, options)) for signal in signals(seed)]
    return Characterisation(
        options=options,
        seed=seed,
        cadences_hz=list(CADENCES_HZ),
        distances_m=list(DISTANCES_M),
        frequencies_hz=list(FREQUENCIES_HZ),
        results=results,
        notes=[
            "the depth filter's cutoff follows range: at the reference distance it is used as "
            "written, nearer it is opened, further it is closed (clamped 0.1-4x)",
            "the lateral filter does not follow range and is characterised at the reference "
            "distance only",
            f"noise is Gaussian with sigma {NOISE_FRACTION_OF_RANGE:.0%} of the distance, "
            "the measured jitter fraction the filter was tuned for; a clean-signal test says "
            "what the filter costs, a noisy one what it buys",
            "these are properties of the filter, not of the estimator: smoothness and delay, "
            "never accuracy",
        ],
    )


def write_characterisation(report: Characterisation, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    as_json = directory / REPORT_FILENAME
    as_json.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    as_csv = directory / TABLE_FILENAME
    with as_csv.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "kind",
                "axis",
                "cadence_hz",
                "distance_m",
                "frequency_hz",
                "samples",
                "rms_raw_m",
                "rms_filtered_m",
                "attenuation_ratio",
                "t90_seconds",
                "settling_seconds",
                "overshoot_fraction",
                "lag_seconds",
                "lag_metres",
                "gain_db",
                "phase_delay_seconds",
            ]
        )
        for row in report.results:
            writer.writerow(
                [
                    row.kind,
                    row.axis,
                    f"{row.cadence_hz:g}",
                    f"{row.distance_m:g}",
                    _cell(row.frequency_hz),
                    row.samples,
                    _cell(row.rms_raw_m),
                    _cell(row.rms_filtered_m),
                    _cell(row.attenuation_ratio),
                    _cell(row.t90_seconds),
                    _cell(row.settling_seconds),
                    _cell(row.overshoot_fraction),
                    _cell(row.lag_seconds),
                    _cell(row.lag_metres),
                    _cell(row.gain_db),
                    _cell(row.phase_delay_seconds),
                ]
            )
    written = [as_json, as_csv]
    for name, figure in (
        (ATTENUATION_FIGURE, _attenuation_figure(report)),
        (STEP_FIGURE, _step_figure(report)),
    ):
        path = directory / name
        figure.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
        _close(figure)
        written.append(path)
    return written


def _attenuation_figure(report: Characterisation) -> Figure:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "unproject-filter"
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    for cadence in report.cadences_hz:
        for distance in report.distances_m:
            rows = [
                r
                for r in report.results
                if r.kind == "sine"
                and r.axis == "depth"
                and r.cadence_hz == cadence
                and r.distance_m == distance
                and r.gain_db is not None
            ]
            if not rows:
                continue
            axes.plot(
                [r.frequency_hz or 0.0 for r in rows],
                [r.gain_db or 0.0 for r in rows],
                marker="o",
                markersize=3,
                linewidth=1.0,
                label=f"{cadence:g} Hz, {distance:g} m",
            )
    axes.set_xscale("log")
    axes.axhline(-3.0, color="#a84848", linestyle="--", linewidth=1.0, label="-3 dB")
    axes.set_xlabel("signal frequency (Hz)")
    axes.set_ylabel("gain (dB)")
    axes.set_title("Depth filter attenuation per frequency, by cadence and distance", fontsize=9)
    axes.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    return figure


def _step_figure(report: Characterisation) -> Figure:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    cadence = report.cadences_hz[1]
    for distance in report.distances_m:
        for signal in signals(report.seed):
            if (
                signal.kind == "step"
                and signal.axis == "depth"
                and signal.cadence_hz == cadence
                and signal.distance_m == distance
            ):
                filtered = run_filter(signal, report.options)
                axes.plot(
                    signal.times - STEP_AT_SECONDS,
                    (filtered - distance) / STEP_METRES,
                    linewidth=1.0,
                    label=f"{distance:g} m",
                )
    axes.axhline(1.0, color="#333333", linewidth=0.8)
    axes.axhline(0.9, color="#cccccc", linewidth=0.8, linestyle="--")
    axes.set_xlim(-0.5, STEP_SECONDS - STEP_AT_SECONDS)
    axes.set_xlabel(f"seconds after a {STEP_METRES:g} m step, sampled at {cadence:g} Hz")
    axes.set_ylabel("response, fraction of the step")
    axes.set_title("Depth filter step response by distance", fontsize=9)
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure


def _close(figure: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)


# --------------------------------------------------------------------------- #
# Parity with the viewer
# --------------------------------------------------------------------------- #

# A small slice of the suite the TypeScript filter is held to. Kept short:
# the fixture is committed and read by both test suites.
PARITY_CADENCES_HZ = (2.5, 25.0)
PARITY_DISTANCES_M = (2.0, 40.0)
PARITY_SAMPLES = 40


def parity_fixture(
    options: Smoothing = DEFAULT_SMOOTHING, *, seed: int = SEED
) -> dict[str, object]:
    """The shared vector: inputs and the outputs this implementation produces."""
    rng = np.random.default_rng(seed)
    cases: list[dict[str, object]] = []
    for cadence in PARITY_CADENCES_HZ:
        for distance in PARITY_DISTANCES_M:
            dt = 1.0 / cadence
            times = np.arange(PARITY_SAMPLES) * dt
            x = 0.3 * np.sin(2.0 * math.pi * 0.5 * times) + rng.normal(0.0, 0.02, PARITY_SAMPLES)
            z = distance + rng.normal(0.0, NOISE_FRACTION_OF_RANGE * distance, PARITY_SAMPLES)
            z[PARITY_SAMPLES // 2 :] += STEP_METRES
            smoother = TrackSmoother(options)
            expected = []
            for time, xi, zi in zip(times, x, z, strict=True):
                [(_, root)] = smoother.frame(float(time), [(0, (float(xi), 0.0, float(zi)))])
                expected.append([round(root[0], 9), round(root[1], 9), round(root[2], 9)])
            cases.append(
                {
                    "cadence_hz": cadence,
                    "distance_m": distance,
                    "samples": [
                        {
                            "time": round(float(t), 6),
                            "x": round(float(a), 6),
                            "y": 0.0,
                            "z": round(float(b), 6),
                        }
                        for t, a, b in zip(times, x, z, strict=True)
                    ],
                    "expected": expected,
                }
            )
    return {
        "comment": (
            "Generated by unproject.rq3.characterise.parity_fixture. Shared between "
            "packages/pipeline (test_rq3_characterise.py) and apps/web (bodies.test.ts): a "
            "noisy step in depth and a noisy sine sideways, at the slowest and fastest cadence "
            "the viewer sees and at the nearest and farthest distance the range term is "
            "clamped over. If either implementation drifts, both suites fail."
        ),
        "options": {
            "lateral": {
                "cutoffHz": options.lateral.cutoff_hz,
                "speedCoupling": options.lateral.speed_coupling,
                "rateCutoffHz": options.lateral.rate_cutoff_hz,
            },
            "depth": {
                "cutoffHz": options.depth.cutoff_hz,
                "speedCoupling": options.depth.speed_coupling,
                "rateCutoffHz": options.depth.rate_cutoff_hz,
            },
            "referenceRangeMetres": options.reference_range_metres,
        },
        "cases": cases,
    }


def write_parity_fixture(path: Path) -> None:
    path.write_text(json.dumps(parity_fixture(), indent=2) + "\n", encoding="utf-8")


def _round(value: float) -> float | None:
    return None if not math.isfinite(value) else round(float(value), 6)


def _cell(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


__all__ = [
    "ATTENUATION_FIGURE",
    "CADENCES_HZ",
    "DISTANCES_M",
    "FREQUENCIES_HZ",
    "REPORT_FILENAME",
    "STEP_FIGURE",
    "TABLE_FILENAME",
    "Characterisation",
    "Signal",
    "SignalResult",
    "characterise",
    "measure",
    "parity_fixture",
    "run_filter",
    "signals",
    "write_characterisation",
    "write_parity_fixture",
]
