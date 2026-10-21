"""The viewer's root filter, in Python, so it can be scored.

This is a second implementation of something that already exists in
`apps/web/src/bodies.ts`, which is normally the thing to avoid. The alternative
was doing the RQ3 analysis in TypeScript, and the analysis belongs where the
rest of it is: deterministic, on the CLI, unit-tested.

The two are held together by a shared fixture rather than by care.
`tests/fixtures/one_euro.json` lists inputs and the outputs both must produce,
and both test suites read it. If they drift, both fail.

What the filter is for: the model places a body about 2.5% of its range away
from where it placed it last, which at 40 m is a metre and reads as people
teleporting. This damps that. It does not make depth more accurate, and the
smoothed numbers are reported beside the raw ones rather than instead of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


class AxisSmoothing(BaseModel):
    """One axis of the filter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Low-pass cutoff while the value is still. Lower is smoother.
    cutoff_hz: float = Field(gt=0.0)
    # How far real movement opens the filter up. Higher is less lag.
    speed_coupling: float = Field(ge=0.0)
    # How quickly the speed estimate itself reacts.
    rate_cutoff_hz: float = Field(gt=0.0)


class Smoothing(BaseModel):
    """The viewer's defaults, restated here as data rather than as constants."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lateral: AxisSmoothing = AxisSmoothing(cutoff_hz=2.5, speed_coupling=0.4, rate_cutoff_hz=1.0)
    depth: AxisSmoothing = AxisSmoothing(cutoff_hz=0.6, speed_coupling=0.0, rate_cutoff_hz=0.25)
    # Filtering follows range, because the error does.
    reference_range_metres: float = Field(default=8.0, gt=0.0)


DEFAULT_SMOOTHING = Smoothing()

# Bounds on the range-proportional term, so one bad depth can neither make the
# filter useless nor freeze it. Mirrors the viewer.
_TIGHTEN_LOW = 0.1
_TIGHTEN_HIGH = 4.0
_NEAREST_RANGE_METRES = 0.5


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class OneEuro:
    """One axis of a One Euro filter, holding its own history."""

    min_cutoff: float
    beta: float
    rate_cutoff: float = 1.0
    value: float | None = None
    rate: float = 0.0

    def filter(self, sample: float, dt: float, scale: float = 1.0) -> float:
        """`scale` weakens the cutoff for this sample; see the range note above."""
        if self.value is None or dt <= 0.0:
            self.value = sample
            return sample
        raw_rate = (sample - self.value) / dt
        self.rate += _alpha(self.rate_cutoff, dt) * (raw_rate - self.rate)
        # The faster it is really moving, the less it is smoothed.
        cutoff = (self.min_cutoff + self.beta * abs(self.rate)) * scale
        self.value += _alpha(cutoff, dt) * (sample - self.value)
        return self.value


class TrackSmoother:
    """Holds each track's root still enough to look like a person."""

    def __init__(self, options: Smoothing = DEFAULT_SMOOTHING) -> None:
        self._options = options
        self._filters: dict[int, tuple[OneEuro, OneEuro, OneEuro]] = {}
        self._last_time: float | None = None

    def frame(
        self, time: float, people: list[tuple[int, tuple[float, float, float]]]
    ) -> list[tuple[int, tuple[float, float, float]]]:
        """Smooth one frame's roots, given `(track_id, (x, y, z))` per person."""
        dt = 0.0 if self._last_time is None else time - self._last_time
        self._last_time = time

        smoothed = [(track, self._one(track, root, dt)) for track, root in people]
        # A track that has gone should not resume from where it left off: the
        # person may be somewhere else entirely by the time the id is reused.
        seen = {track for track, _ in people}
        for track in [key for key in self._filters if key not in seen]:
            del self._filters[track]
        return smoothed

    def _one(
        self, track: int, root: tuple[float, float, float], dt: float
    ) -> tuple[float, float, float]:
        axes = self._filters.get(track)
        if axes is None:
            lateral, depth = self._options.lateral, self._options.depth
            axes = (
                OneEuro(lateral.cutoff_hz, lateral.speed_coupling, lateral.rate_cutoff_hz),
                OneEuro(lateral.cutoff_hz, lateral.speed_coupling, lateral.rate_cutoff_hz),
                OneEuro(depth.cutoff_hz, depth.speed_coupling, depth.rate_cutoff_hz),
            )
            self._filters[track] = axes

        raw_x, raw_y, raw_z = root
        # Proportional both ways: someone at two metres is barely noisy and
        # should barely be filtered.
        tighten = min(
            _TIGHTEN_HIGH,
            max(
                _TIGHTEN_LOW,
                self._options.reference_range_metres / max(abs(raw_z), _NEAREST_RANGE_METRES),
            ),
        )
        along_x, along_y, along_z = axes
        # Only depth follows the range; sideways is accurate at any distance.
        return (
            along_x.filter(raw_x, dt),
            along_y.filter(raw_y, dt),
            along_z.filter(raw_z, dt, tighten),
        )
