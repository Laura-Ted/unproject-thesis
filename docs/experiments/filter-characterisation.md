# Temporal Filter Characterisation

**Status:** Frozen
**Created:** 2026-09-15
**Frozen:** 2026-09-15, with the commit that adds it
**Governs:** the smoothing claims of RQ3
(`docs/adr/ADR-0002-live-body-capture-from-monocular-video.md` §4), as
required by `docs/adr/ADR-0005-thesis-evidence-hardening.md` WP6

The viewer's root filter - a One Euro filter per axis, with the depth cutoff
following range - is scored on real footage for the apparent depth velocity it
removes. That is a measure of **smoothness**, and the ADR's §3 says it may be
called nothing else: the footage has no truth to be accurate against, and a
smoother trajectory is not a more correct one.

This protocol gives the filter a truth. A deterministic suite of synthetic
root trajectories, each with a known clean value, runs through the same
`TrackSmoother` the analysis uses, and the filter is measured for what a
filter is measured for. The outcome is the stability/delay trade-off as
numbers, on a signal whose answer is known.

## 1. Signals

Built from seed 20260915 by `unproject.rq3.characterise.signals`, in a fixed
order, for the depth axis at every combination below and for the lateral axis
at the reference distance:

| | Values |
|---|---|
| Sample cadence | 2.5, 10, 25 samples/s - a camera answered at the estimator's rate, a clip at stride 2, a 25 fps clip at stride 1 |
| Distance | 2, 8, 20, 40 m - arm's length, the filter's reference, mid and far crowd; the range term is clamped over this span |
| Constant with noise | 400 samples, Gaussian noise with σ = 3 % of the distance, the jitter fraction measured on the crowd clip |
| Step | 1 m further, 2 s in, 10 s long |
| Ramp | 1 m/s away, 8 s |
| Sinusoid | 0.5 m amplitude at 0.1, 0.25, 0.5, 1, 2 Hz, four periods, only below the cadence's Nyquist limit |

The lateral filter does not follow range and is run at 8 m only.

## 2. Metrics

| Signal | Metric | Definition |
|---|---|---|
| noise | RMS before and after | root-mean-square distance from the clean value, raw and filtered; their ratio is the attenuation |
| step | t90 | seconds from the step to the first sample within 90 % of it |
| step | settling | seconds to the first sample after which every sample stays within 10 % of the step |
| step | overshoot | the largest excursion beyond the step, as a fraction of it |
| ramp | lag | how far behind the truth the filtered value runs over the last second, in metres and, over the slope, in seconds |
| sinusoid | gain | output over input amplitude at the driving frequency, in dB, from a sine-and-cosine fit after the first period |
| sinusoid | phase delay | the fitted phase, converted to seconds |

## 3. Parity with the viewer

The filter exists twice, in Python for the analysis and in TypeScript for the
viewer. `packages/pipeline/tests/fixtures/one_euro_signals.json` is a slice of
this suite - a noisy step in depth and a noisy sine sideways at 2.5 and 25
samples/s, at 2 and 40 m - with the Python outputs recorded. The Python test
checks the committed file is what the code generates and that the filter
reproduces it; the viewer's test replays the same inputs through its own
filter and requires the same outputs to 1e-5 m. Either drifting fails both.
It joins `one_euro.json`, which covers the defaults at one cadence.

## 4. Commands and outputs

```shell
uv run unproject filter-characterise          # writes artifacts/filter/
```

| Output | Contents |
|---|---|
| `filter-characterisation.json` | the options, the seed, every signal's metrics |
| `filter-characterisation.csv` | one row per signal |
| `filter-attenuation.svg` | depth gain per frequency, by cadence and distance |
| `filter-step-response.svg` | depth step response by distance at 10 samples/s |

No inference, no data, deterministic: the same command regenerates the same
bytes.

## 5. What this does and does not say

It says what the filter costs and buys: how much of a known noise it removes,
how long it takes to admit that something moved, how far behind a walker it
runs, and which frequencies of real motion it attenuates. Those are
properties of the filter, reported so the thesis can state the trade-off with
numbers rather than adjectives.

It does not say the filtered trajectory is closer to where the person was.
On MuPoTS the raw and smoothed metrics are computed over the same pairs and
reported side by side (`ground-truth-protocol.md` §4.3); on the crowd clips
the filter is an ecological control of smoothness only.
