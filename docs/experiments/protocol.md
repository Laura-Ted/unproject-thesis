# Measurement Protocol

> **RQ1/RQ2 superseded for the final results on 2026-09-15.** The
> measurements taken under §2–§7 of this document are the **pilot**: ADR-0005
> §2.3 records why they cannot be the sole basis of the conclusions - an
> unrecorded source, the same forty frames in one process, a timer that
> excluded serialisation, dirty manifests. The final RQ1/RQ2 protocol is
> `docs/experiments/protocol-v2.md`. Nothing below is altered; it stays the
> record of what the pilot measured and how.

**Status:** Frozen
**Created:** 2026-09-04
**Frozen:** 2026-09-05, before any measurement recorded under it
**Governs:** RQ1, RQ2 and RQ3 of
`docs/adr/ADR-0002-live-body-capture-from-monocular-video.md`

This is the pre-registered protocol. It was frozen before the first measured run
and nothing in it was chosen after seeing a result.

It contains no results and none may be added: raw observations live with their
runs, as machine-readable output, and every table and figure is derived from
those by `unproject benchmark-summary` and `unproject rq3`.

Changing this document now requires a dated note in `docs/notes/` saying what
changed, why, and which results were collected under which version. Two sections
declare work this protocol governs but that has **not been carried out** - §2.3
and §3.1 - and they say so in place rather than being quietly dropped.

---

## 1. Environment

The measuring machine is the development machine:

| | |
|---|---|
| Machine | Apple MacBook Pro, M3 Pro, 18 GB unified memory |
| OS | macOS 26.6.2 (build 25G83) |
| Python | 3.12, pinned in `.python-version` |
| Backend | PyTorch MPS; a CPU fallback must be labelled as such |

This resolves the M3 Pro / M4 Pro discrepancy recorded in ADR-0002 §5.1 in
favour of the M3 Pro. See `docs/notes/2026-09-04-wp0-documentation-consolidation.md`.

Every run additionally records its own environment in its manifest, per ADR-0002
§7.5. The table above is the declared target, not a substitute for that record.

## 2. Scenes

Each scene declares its source, why it was chosen, its consent status, and which
research question it serves.

### 2.1 The crowd clip, and three controlled transformations of it

| | |
|---|---|
| Source | `datasets/crowd/clip.mp4`, 1920x1080, 460 frames at 23.976 fps |
| Content | A locked-off shot of people walking through a park |
| Serves | RQ1, and the aspect-ratio, resolution and output factors of RQ2 |
| Consent | **Unresolved.** See §8. |

The other three cells of the matrix are transformations of that one file, so the
content is held as close to constant as a transformation allows. Produced with:

```shell
ffmpeg -y -i datasets/crowd/clip.mp4 -vf "scale=960:540" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an datasets/crowd-half/clip.mp4
ffmpeg -y -i datasets/crowd/clip.mp4 -vf "crop=1080:1080:420:0" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an datasets/crowd-square/clip.mp4
ffmpeg -y -i datasets/crowd/clip.mp4 -vf "crop=1080:1080:420:0,scale=540:540" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an datasets/crowd-square-half/clip.mp4
```

Encoder settings are identical across all three so the transformation is the only
difference. The pipeline decodes to lossless PNG before the model sees anything,
so the re-encode affects what is decoded but not how it is stored.

**The square crop is a confounded comparison, and is declared so here rather
than discovered later.** Cropping 1920 wide to 1080 removes part of the scene,
so the square cells show fewer people than the 16:9 ones. The number of people
per frame is recorded in every benchmark result, so the confounder is visible in
the data rather than only acknowledged in prose. Any statement about aspect ratio
drawn from these cells is observational.

### 2.2 The fixture clip

`datasets/fixture/clip.mp4` is a synthetic `testsrc2` pattern with nobody in it.
It exercises the plumbing and produces no thesis number. Consent does not arise.

### 2.3 Scenes for the person-count factor - NOT COLLECTED

RQ2 asks for low, medium and high visible-person counts. That needs three scenes
with different numbers of people in them, and only one scene with people in it
exists in this repository. **No footage was recorded for this factor and it is
not measured.** See §3.1.

### The RQ3 controlled scene

> **Superseded.** ADR-0004 replaces this with evaluation on MuPoTS-3D and
> Anny-One. No participant recording, tape measurement or floor marker is
> required for the thesis. See `docs/experiments/ground-truth-protocol.md`. What
> follows is kept as the record of the plan it replaces.

Shape fixed as of WP5; the recording itself has not been made. Per ADR-0002 §8.4
it is the smallest scene that answers RQ3, and it is a case study rather than an
accuracy benchmark:

- one fixed camera, settings unchanged for the whole session, height and
  placement uncertainty recorded;
- floor markers at several lateral and depth positions, measured with a tape
  from directly under the camera, each with its own uncertainty. Positions are
  in the artifact's own frame, which avoids needing a camera pose nothing here
  estimates, and limits the method to a single fixed camera;
- one consenting participant. A second only for a declared occlusion or tracking
  question;
- stationary intervals at known markers, long enough to have samples in them;
- one simple measured walking path;
- annotation boundaries read off the footage, with their own uncertainty
  recorded per interval.

The annotation file format is `experiments/rq3/corridor.example.toml`, which is
a template and contains no measurements. `unproject validate-config` checks it.

## 3. Variables

RQ2 varies, per ADR-0002 §8.3, only these:

| Factor | Levels | Executed |
|---|---|---|
| Visible people | low, medium, high | **No** - see §3.1 |
| Aspect ratio | 16:9, square | Yes, observational - see §2.1 |
| Resolution | 1920 and 960 long edge (16:9), 1080 and 540 (square) | Yes |
| Output mode | skeleton, mesh | Yes |

The three executed factors are fully crossed: eight cells, one benchmark config
each in `experiments/benchmark/`, named `<scene>-<output>.toml`. Every cell uses
the same warm-up, measured-frame count and repetitions; only the factors differ.

### 3.1 The person-count factor is not measured

Only one scene with people in it exists here. Measuring person count would need
two more, and no footage was recorded for them.

This is not a gap that can be papered over by counting the people the model
happens to find in the one clip: that varies with the clip's own content rather
than being set by the experiment, so it is an observation about this footage and
not a level of a factor. `people_median` is recorded in every result and may be
reported as a description of the scene. **No claim about how cost scales with
the number of people may be drawn from this protocol.**

## 4. Warm-up and repetition

Identical in every cell, fixed here before collection:

| | |
|---|---|
| Warm-up frames | 5, discarded |
| Measured frames | 40 |
| Repetitions | 3, each opening a fresh session |
| Steady-state samples per cell | 120 |

40 frames of a 230-frame ingest, taken from the start, so every cell sees the
same part of the clip. Three repetitions is the minimum ADR-0002 §8.2 asks for
and the most that fits the runtime: eight cells at roughly 0.4 s a frame is
already about ten minutes of measurement per full pass.

Cold start is the time to build the model and open a session, recorded per
repetition and never averaged into the per-frame figures. Only repetition 0 of
each invocation is genuinely cold; the two after it reuse the loaded model and
are reported separately as warm starts.

Individual observations are preserved. Every frame that was timed is in the raw
result file, not just its summary.

## 5. Metrics

### RQ1 and RQ2

Recorded by `unproject benchmark` into the result schema, and summarised by
`unproject benchmark-summary`. Seconds unless stated:

| Metric | Definition |
|---|---|
| Per-frame latency | Wall-clock around one `session.frame` call, on a monotonic high-resolution clock. In mesh cells it includes building a track's unposed body the first time that person is seen, which is what the socket pays. |
| Median, p90, p95, min, max | Over the steady-state samples of a cell. Percentiles are nearest-rank, so a reported value is one that was measured. |
| Median frames per second | The reciprocal of the median frame, not a total divided by a duration: a mean rate over a run that stalled once describes neither half of it. |
| Cold start | Building the model and opening a session, repetition 0 only. |
| Warm start | The same call in repetitions 1 and 2, with the model already built. |
| Peak process memory | Whole-process high-water mark, in bytes. Not this stage's own allocation. |
| Metal allocated and driver memory | `torch.mps` counters where the installed PyTorch exposes them, in bytes; absent otherwise rather than zero. |
| People per frame | Detections above the configured confidence threshold. A description of the scene, not a factor - see §3.1. |
| Input geometry | Width and height of each frame as measured, so a cell cannot be mislabelled. |

The end-to-end transport overhead named in the README - the cost of the socket
on top of inference - is **not measured by this protocol.** It needs the browser,
and ADR-0002 §12 keeps the browser out of the benchmark except for an explicit
end-to-end test that is not part of this work.

### RQ3

Implemented and fixed as of WP5. Everything below is metres and seconds in
camera-relative OpenCV axes (+X right, +Y down, +Z forward), computed by
`unproject rq3` from a bodies artifact and an annotation file:

| Metric | Definition |
|---|---|
| Lateral and depth position error | Median signed and absolute difference between the estimated root and the annotated marker, per axis, over a stationary interval. Reported beside the marker's own uncertainty. |
| Apparent velocity by axis | Median and p95 of \|Δx\|/Δt and \|Δz\|/Δt between consecutive detections of one track. Pairs more than 0.5 s apart are counted as gaps and excluded. |
| Depth jitter as a fraction of range | Median of \|Δz\| divided by the range at that frame, which is the form the error actually takes. |
| Within-track shape variation | Standard deviation of each shape parameter within a track; the maximum and mean across parameters are reported. |
| Track fragmentation | Track count against participant count, median track lifetime, and the longest hole inside a track. |
| 2D reprojection consistency | Fraction of detections whose root, projected through that frame's recorded intrinsics, lands inside its own recorded 2D box, and the median pixel distance for those that do not. **Reported separately for boxes fully inside the frame and boxes clipped by its edge; only the first is interpretable.** A body walking out of the bottom of the shot has its legs cut off, so its box stops at the image edge while its pelvis is genuinely below it. |
| Whole-run motion and smoothing | The velocity, jitter and smoothing metrics above, over every detection in a run rather than over an annotated interval. Needs no ground truth, so it is what an uncontrolled scene can produce. |
| Response delay | Reported only for a single continuous track. Over interleaved tracks the raw and smoothed sequences are not one signal and the best-aligning shift means nothing. |
| Smoothing effect | Every motion metric above, recomputed on the filtered roots and reported beside the raw ones, plus the response delay. |
| Response delay definition | The whole-frame shift that best aligns the smoothed depth signal with the raw one, in seconds. |

Percentiles are nearest-rank, matching the benchmark, so a reported value is one
that was measured.

**Not interpolated.** A frame where nobody was found contributes nothing. No
metric fills a gap in, and the count of skipped pairs is reported rather than
absorbed.

## 6. Failure and exclusion rules

Written before collection:

- A run that raises is recorded with `status: "failed"` and its error, and the
  frames it did manage to time stay in the file. It is counted in `failed_runs`
  and its error is printed by the summary.
- A dry run measures nothing and is excluded from every aggregate. The exclusion
  is counted and reported, so a directory of dry runs cannot read as an empty
  one.
- A result file that stops parsing is named in the summary's errors rather than
  skipped.
- **No run is excluded for being surprising.** The only grounds for excluding a
  completed run from a reported figure are a recorded failure, a dry run, or a
  documented environmental fault written up in `docs/notes/` on the day. Nothing
  is dropped after the fact for disagreeing with an expectation.
- A cell whose steady-state sample count is not 120 is reported with its actual
  count rather than topped up.

## 7. Aggregation, tables and figures

Aggregation is deterministic, runs no inference, and reads only raw
observations. Two commands regenerate everything:

```shell
uv run unproject benchmark-summary                       # RQ1, RQ2
uv run unproject rq3 <bodies artifact dir> <annotations> # RQ3
```

Planned outputs, listed before collection so the analysis cannot grow to fit the
result:

| Output | Produced by | Contents |
|---|---|---|
| `artifacts/benchmark/summary.json` | `benchmark-summary` | Per cell: sample count, median, p90, p95, min, max, median fps, cold and warm starts, peak and Metal memory, people median, input geometry, source ingest hashes, errors |
| `artifacts/benchmark/summary.csv` | `benchmark-summary` | The same, one row a cell, for inspection and thesis tooling |
| `rq3.json` | `rq3` | Every RQ3 metric in §5, raw and smoothed |
| `rq3-positions.csv` | `rq3` | Position error per marker with the marker's own uncertainty in its own column |
| `rq3-position-error.svg` | `rq3` | Lateral against depth error per marker, with the marker uncertainty drawn on |
| `rq3-axis-velocity.svg` | `rq3` | Apparent velocity per axis, raw against smoothed, response delay in the title |
| `rq3-depth-jitter.svg` | `rq3` | Depth jitter as a percentage of range, per interval |

No other table or figure is planned. One added later is a post-hoc analysis and
is labelled as such.

## 8. Privacy and consent

**Unresolved for the crowd clip.** `datasets/crowd/clip.mp4` shows identifiable
people in a public place and this repository records nothing about where it came
from or on what terms. It is gitignored and nothing about it leaves the machine,
which satisfies the storage and processing constraints. It does not settle
whether the footage may be used in a submitted thesis, or whether a frame from
it may be published as a figure. **That has to be answered before any figure
drawn from this clip appears in the thesis**, and the answer belongs in this
section.

Per ADR-0002 §5.2: no identifiable footage is committed, nothing leaves the
machine, participants consent before controlled recording, and only the footage
the declared experiment needs is collected. The consent wording itself is _to be
decided_ before the RQ3 case study is recorded.
