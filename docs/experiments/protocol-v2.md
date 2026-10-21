# Measurement Protocol v2 — RQ1 and RQ2

**Status:** Frozen
**Created:** 2026-09-15
**Frozen:** 2026-09-15, with the commit that adds it, before any measurement
recorded under it
**Governs:** RQ1 and RQ2 of `docs/adr/ADR-0002-live-body-capture-from-monocular-video.md`,
as required by `docs/adr/ADR-0005-thesis-evidence-hardening.md` WP2
**Supersedes for the final results:** the RQ1/RQ2 part of
`docs/experiments/protocol.md`, which is left as written and whose measurements
are the **pilot**. Nothing in v1 is altered retroactively.
**Amended:** 2026-09-15 by ADR-0006, before any final measurement under this
protocol: §2 names a set of declared machines instead of one, §4.2 pre-registers
the person-count suite that §4.1 left unmeasured, §10 adds the CPU-only point,
§11 the cross-machine reading rules. What changed and why is in
`docs/notes/2026-09-15-adr0006-campaign.md`; the pilot on the M4 of the same
day was taken under the text as first frozen and is a pilot either way.
**Erratum:** 2026-09-23, after collection, wording only: §6 read as if the
median were nearest-rank. It is not, and never was; no result changes. See
§6 and `docs/notes/2026-09-23-code-cleanup.md`.

This is the pre-registered protocol for the final performance evidence. It
contains no results and none may be added: raw observations live with their
runs, as machine-readable files, and every table and figure is derived from
them by `unproject benchmark-suite-summary`.

Changing this document after a measurement has been taken under it requires a
dated note in `docs/notes/` saying what changed, why, and which results were
collected under which version.

---

## 1. Why a second protocol

The v1 matrix answered RQ1 and RQ2 well enough to steer the project and not
well enough to close it. ADR-0005 §2.3 lists the reasons; each is a design
decision below.

| v1 | v2 |
|---|---|
| `datasets/crowd/clip.mp4`, provenance unrecorded, consent unresolved | a CC0 clip with source, author, licence, URL, date and digest in the experiment file |
| the first 40 frames of the clip | 60 frames spread along the whole clip, the same frames in every cell |
| 3 repetitions in one process: the first cold, the rest warm | 3 repetitions, each in a fresh process: all cold, nothing carried over |
| `session.frame` on a file path, disk read included; serialisation excluded | estimator on a frame already in memory; rest bodies and the wire message timed apart |
| memory as a whole-process high-water mark shared across repetitions | baseline, after-load and peak per process; Metal baseline and peak |
| throughput as the reciprocal of the median | frames over the wall clock of the steady phase, per job |
| square cells by centre crop, which removed people | square cells by letterbox, same content, same apparent scale |
| cells run in file order | jobs shuffled with a recorded seed, repetitions interleaved |
| every manifest `git_dirty=true` | every result records commit and dirty state; the final run is from a clean tree |

## 2. Environment

*Amended by ADR-0006 (2026-09-15).* The final results are a **campaign**: the
same suites, from the same commit, on a set of machines declared before they
measure in `benchmarks/machines.toml`. Each machine has an id that every job
records (`environment.machine_id`), an accelerator it is expected to offer,
and a hardware class the thesis counts it as. The set at the amendment:

| Id | Machine | Backend | Class |
|---|---|---|---|
| A | Apple MacBook Pro, M3 Pro, 18 GB unified memory - the machine of the first freeze | PyTorch MPS | consumer |
| B | Apple Mac mini, M4, 16 GB unified memory | PyTorch MPS | consumer |
| C-* | Linux, one NVIDIA GeForce card (the id names it), rented by the hour with the card in passthrough | PyTorch CUDA | consumer silicon on a shared host, declared as such |

A machine is added to the registry before it runs and never after; a result
from an id that is not in the registry is a pilot. Two machines never write
under one artifacts root (`make bench-* MACHINE=<id>` gives each its own), and
the summary marks a cell whose jobs name two ids or two devices as not
meeting the protocol.

Common to every machine: Python 3.12 pinned in `.python-version`; the
environment from the lockfile (`uv sync --frozen`), so torch, numpy and the
estimator are the same versions everywhere; the device chosen as CUDA, then
Metal, then CPU, or the one the suite insists on, and never a silent
fallback - a job that cannot get the device it was asked for fails and says
so (`unproject.bodies.device`). Every job records the chip, OS release,
physical memory, Python, package versions, device kind and name, CUDA and
driver versions where there are any, torch's thread count, commit and dirty
state it actually ran under.

Manual conditions are recorded per job, before and after, as the OS reports
them: on macOS the power source, low-power mode and the thermal CPU speed
limit; on Linux the CPU frequency governor and the card's temperature, power
draw, clocks, performance state, utilisation and throttle reasons from
`nvidia-smi`; on both the one-minute load average. The final run on every
machine is on mains, low-power mode off, no other foreground work and nothing
else on the accelerator, and `unproject benchmark-preflight` refuses to start
a suite on a dirty tree, an undeclared machine, the wrong device, footage
whose bytes are not the campaign's, a full disk or a card another process is
using. A job whose CPU limit or throttle reasons changed mid-run is reported
with that fact beside it.

## 3. Scenes

One public sequence and three controlled transformations of it. The content
is the same in all four; nothing is cropped away.

> **Amended 2026-09-15 (evening), before collection.** The campaign's
> matrix runs on **MOT16-09** (MOTChallenge), not on the pedestrians clip
> below, and its three transformations are made **by the ingest** from the
> official JPEG frames, not by ffmpeg: `experiments/mot16-09-half.toml`
> (`geometry.downscale = 2`, an exact integer box average over 2×2 blocks),
> `mot16-09-letterbox.toml` (`geometry.letterbox_square = true`, 420 black
> rows above and below) and `mot16-09-letterbox-half.toml` (both). The
> table of §3.2 applies with `mot16-09` in place of `pedestrians` and the
> same source and tensor sizes. Two reasons, both settled before any job
> ran: the campaign's inputs are the two research datasets (MOT16,
> MuPoTS-3D) and nothing else; and integer arithmetic on the same source
> bytes gives every machine the same frames, so nothing has to be transcoded
> once and carried, and a third party reproduces the bytes from the archive
> and the config alone. Sections 3.1-3.3 below describe the pedestrians
> clip as the pilot of 2026-09-15 morning used it, and remain the record of
> that pilot; the clip stays in the thesis for the RQ3 temporal analysis.
> The suite files are `matrix-suite.toml` (`rq1rq2-v2-matrix`) and
> `cpu-suite.toml` (`rq1-v2-cpu`); the pedestrians ones are retired.

### 3.1 The source

| | |
|---|---|
| Title | Video Codec Test: pedestrian area, 1080p25 |
| Source | <https://commons.wikimedia.org/wiki/File:Video_Codec_Test_pedestrian_area_1080p25.y4m.webm> |
| Licence | CC0 1.0 Universal, public domain dedication |
| Content | a pedestrian street in Munich, stationary camera, about ten people, whole bodies in frame |
| Original | 11,215,394 bytes, webm, sha256 `bfadaa62cccb42db875d50bb842aa0964fbf72040432e4097c1df59e043e0c26` |
| Acquired | 2026-09-06 |
| Decoded | 375 frames, 1920×1080, 25 fps, stride 1 |

Every fact above is also in `experiments/pedestrians.toml` under
`[provenance]`, and every job result embeds it. The acquisition and transcode
commands are in that file and in `docs/reproduce.md` §2.

### 3.2 The transformations

Produced with ffmpeg from `datasets/pedestrians/clip.mp4`, identical encoder
settings, so the transformation is the only difference:

```shell
ffmpeg -y -i datasets/pedestrians/clip.mp4 -vf "scale=960:540" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an datasets/pedestrians-half/clip.mp4
ffmpeg -y -i datasets/pedestrians/clip.mp4 -vf "pad=1920:1920:0:420:color=black" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an datasets/pedestrians-letterbox/clip.mp4
ffmpeg -y -i datasets/pedestrians/clip.mp4 -vf "pad=1920:1920:0:420:color=black,scale=960:960" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an datasets/pedestrians-letterbox-half/clip.mp4
```

| Scene | Source size | Tensor the model sees | What changed |
|---|---|---|---|
| `pedestrians` | 1920×1080 | 768×432 | nothing |
| `pedestrians-half` | 960×540 | 768×432 | half the source pixels; the same tensor |
| `pedestrians-letterbox` | 1920×1920 | 768×768 | 336 black rows in the tensor; same content, same apparent scale |
| `pedestrians-letterbox-half` | 960×960 | 768×768 | both |

**Letterbox, not crop.** v1's square cells were centre crops, which removed
part of the scene and half the people, so aspect ratio and content changed
together. Padding keeps every person in every cell.

**One transcode, carried.** *(ADR-0006.)* libx264 is not bit-stable across
ffmpeg builds, so a machine that re-ran the commands above would measure
slightly different pixels. The campaign's bytes are the four files as made
once on machine B (ffmpeg 9.0.1, 2026-09-15), whose digests are pinned as
`expected_sha256` in the four experiment files; they are carried to every
machine as they are, deposited with the evidence, and the preflight refuses a
file that hashes differently. The commands stay as the recipe that produced
them.

**Residual apparent-scale variation: none for the aspect factor.** Multi-HMR
resizes the long edge to 768 whatever the source, and the long edge is 1920 in
both the 16:9 and the letterboxed clip, so a person covers the same pixels of
the tensor in both; the tensor differs only by the padded rows, which is the
factor under test. The resolution factor changes the *source* detail, not the
tensor: a 960-wide source is upsampled to the same 768×432. That factor
therefore measures decoding and resampling, and whatever detail survives, not
a smaller model input - which is what the v1 result "resolution barely
matters" already showed and could not explain.

The decoded frames are lossless PNG in the ingest artifact, so the transcode
decides what is decoded and not how it is stored.

### 3.3 Consent

CC0 1.0: nothing about using the clip, or a frame of it in the thesis, is in
question. The crowd clip of v1 is not used.

## 4. Variables

Per ADR-0002 §8.3, and only these:

| Factor | Levels | Executed |
|---|---|---|
| Aspect ratio | 16:9, 1:1 by letterbox | yes, controlled |
| Resolution | 1920 and 960 long edge | yes, controlled |
| Output mode | skeleton, mesh | yes |
| Visible people | low, medium, high | **no** - one scene; see §4.1 |

The three executed factors are fully crossed: eight cells, in
`experiments/benchmark/v2/matrix-suite.toml`, named `<scene>-<output>`.

### 4.1 The person-count factor is not measured by the matrix suite

One scene with people in it. People per frame is recorded in every sample and
may be reported as a description of the scene and used as a covariate in a
figure. **No claim about how cost scales with the number of people may be
drawn from the matrix suite**, and no correlation with it is a cause.

### 4.2 The person-count suite

*Added by ADR-0006 (2026-09-15), before collection.* The factor is measured
by a second suite, `experiments/benchmark/v2/people-suite.toml`, under the
same policy (§5) and metrics (§6): every scene is 1920×1080 so the model
sees the same 768×432 tensor in every cell, and the scenes are chosen for
how many people are in them, all from a fixed camera at eye level with whole
bodies in frame and no vehicles:

| Level | Scene | Source | People per frame (pilot count, median) |
|---|---|---|---|
| low | `mupots-ts10` | MuPoTS-3D TS10, three people walking towards the camera on a campus courtyard | 3 |
| low | `mupots-ts11` | MuPoTS-3D TS11, two people walking and crossing, same campus | 2 |
| medium | `mot16-09` | MOTChallenge MOT16-09, shop front on a pedestrian street; the matrix suite's sequence | about 11 |
| high | `mot16-08` | MOTChallenge MOT16-08, pedestrian shopping street | about 23 |

*Amended 2026-09-15, still before collection:* the two MuPoTS-3D sequences
were added as the low level after a search of the public datasets found no
fixed-camera 1080p footage of two or three people walking with a free
licence; MuPoTS-3D is already the dataset of RQ3, its terms are already
accepted, and its camera was measured still on every sequence. The author's
own recording (`walkers`), pre-registered earlier the same day as the low
scene, was dropped the same evening: the two MuPoTS sequences carry the
level, and no recording will be made. The pedestrians clip, the second
medium scene of the first draft, left the suite the same evening with the
decision that the campaign's inputs are the two research datasets only. No
job of the suite had run when any of these changes was made; the suite is
the four scenes above and is not extended after collection starts.

Two scenes at the low level on purpose: their agreement or disagreement is
the check on whether the other pairs read the count or the scene. Output mode is crossed with the scene (skeleton and mesh), because
per-person work is where the two differ. The suite file is final; a run
of a suite with a different set of cells is a run of a different suite (the
hash says so).

**Reading rule, fixed here.** The scenes are different clips, so no pair
times the same frames (`same_frames: false`) and the paired reading of §6
does not apply; the reading is the difference of cell medians and, within
every cell, the per-frame inference time against the number of people the
model returned for that frame. The pre-registered expectation is a cost that
is flat in the backbone and grows in the per-person head: a positive slope
within and across scenes, small against the tensor effect of §3.2. The count
remains a covariate: the scenes differ in more than their count, and the
slope is reported as an observation, not as a cause.

## 5. Measurement policy

Identical in every cell, fixed here before collection:

| | |
|---|---|
| Warm-up frames | 5, the first five of the clip, timed and marked, never aggregated |
| Measured frames | 60, spread evenly over frames 5–374, the same list in every cell and repetition |
| Repetitions | 3, each a fresh interpreter process |
| Order | all 24 (cell, repetition) jobs shuffled with `order_seed = 20260915`, repetitions interleaved; the index of every job is in its result |
| Per process | model load, session open, warm-up, steady state, in that order, once |

Frames are decoded from the ingest artifact into memory before the clock
starts. The frame order is monotone, because the tracker expects frames in the
order they were shot.

## 6. Metrics

Recorded by `unproject benchmark-job` into the result schema
(`unproject.benchmark.results`, schema 2), one file per job, and summarised by
`unproject benchmark-suite-summary`. Seconds unless stated. All percentiles are
nearest-rank, so a reported value is one that was measured.

> **Erratum 2026-09-23, wording only.** The median is not one of those
> percentiles: it is the middle value, or the mean of the two middle values
> for an even count - every cell has 3 × 60 = 180 samples, so every median is
> such a mean - as `unproject.benchmark.summary` computes it and as the thesis
> tables do (`numpy.median`). p25, p75, p90 and p95 are nearest-rank, as
> stated. Every result was computed this way; nothing is recomputed.

| Metric | Definition |
|---|---|
| Model load | First `open()` of the estimator in the process: import, checkpoint digest, build. Once per job; never averaged into frames. |
| Session open | Second `open()`: a tracker over the loaded model. |
| Inference | `session.frame` on an in-memory RGB array: upstream preprocessing, forward pass, tracker, conversion to the project's types. |
| Rest bodies | Mesh cells only: the unposed body of each person seen for the first time, which the socket sends once per track. Zero in skeleton cells. |
| Serialisation | The wire message built by `unproject.bodies.wire.frame_message` and encoded to JSON: the same code the socket runs. Its byte size is recorded. |
| Total | Inference + rest bodies + serialisation, per frame. |
| Median, p25, p75, p90, p95, min, max | Over the steady-state samples of a cell, for inference and for total. IQR is p75 − p25. |
| Throughput | Steady frames divided by the wall clock from the start of the first steady frame to the end of the last one's serialisation, **per job**; the median of the three is the cell's. |
| RSS baseline / after load / end / peak | Resident set size before the model is built, after, at the end, and the process high-water mark. Model footprint is after-load minus baseline. Bytes. |
| Metal allocated / driver | `torch.mps` counters after load (baseline) and their maximum over the frames (peak). Absent, not zero, where unavailable. |
| People per frame | Detections above the configured threshold. A covariate, not a factor. |
| Input geometry | Source width and height, and the padded tensor the model sees. |
| Conditions | Power source, low-power mode, CPU speed limit before and after. |
| Paired difference | For a declared pair of cells, `b − a` of the total per frame on the same repetition and position; median, p25, p75, the fraction of frames where b was slower, and the ratio of medians. Only valid where both cells timed the same source frames, which the summary checks. |

## 7. Failure and exclusion rules

Written before collection:

- A job whose process exits non-zero is recorded in the run record with its
  exit code; if it wrote a result before dying, that result carries
  `status: "failed"` and the frames it managed, and is excluded from every
  aggregate. Counted in `failed_jobs`; listed in `errors`.
- A dry run measures nothing and is excluded from every aggregate. The
  exclusion is counted.
- A result that stops parsing is named in the summary rather than skipped.
- A cell with fewer than three complete jobs, fewer than sixty steady frames
  in any of them, or jobs that timed different frame lists is reported with
  `meets_protocol: false`. It is never topped up.
- A job whose CPU speed limit changed between start and end is reported with
  both values; it is not excluded, and the reader decides.
- **No job is excluded for being surprising.** The only grounds are the ones
  above and a documented environmental fault written up in `docs/notes/` on
  the day.

## 8. Aggregation, tables and figures

Deterministic, runs no inference, reads only raw results:

```shell
uv run unproject benchmark-suite experiments/benchmark/v2/matrix-suite.toml
uv run unproject benchmark-suite-summary experiments/benchmark/v2/matrix-suite.toml
```

Planned outputs, listed before collection:

| Output | Contents |
|---|---|
| `<root>/benchmark-v2/rq1rq2-v2-matrix/<cell>/<stamp>-rep<k>-job<i>.json` | One job: every frame timed, phases apart, memory, conditions, geometry, provenance, environment, git state. `<root>` is `artifacts/campaign/<machine id>` in the campaign (§2), `artifacts` for a run outside it |
| `.../preflight-<stamp>.json` | What `benchmark-preflight` found on the machine before the suite ran (ADR-0006) |
| `.../<stamp>-suite-run.json` | The driver's record: order, seed, exit code and result file per job |
| `.../summary.json` | Per cell: every metric in §6; per pair: the paired difference |
| `.../summary.csv` | One row per cell |
| `.../pairs.csv` | One row per pair |

The thesis figures for RQ1/RQ2 are drawn from `summary.json` only (WP7). No
other table or figure is planned; one added later is post-hoc and labelled so.

## 9. What this protocol does not measure

Ingest, the WebSocket, and the browser. Those are the live path and are
measured under WP3 with their own definitions; nothing here is an end-to-end
number and none may be presented as one.

## 10. The CPU-only point

*Added by ADR-0006 (2026-09-15), before collection.* RQ1 asks what consumer
hardware can do, and a consumer machine without a usable accelerator is one
answer. `experiments/benchmark/v2/cpu-suite.toml` runs the two 16:9 cells
of the matrix - skeleton and mesh on MOT16-09 (since 2026-09-15 evening,
before collection; the first draft named the pedestrians clip) - under the
same policy with `device = "cpu"`, which the driver and every job enforce. Two cells, not eight: the tensor and resolution factors are
properties of the model that the matrix measures, and a CPU cell costs about
ten times an accelerated one. Torch's thread count is recorded and not tuned;
the number is what the default install does. On a rented host the CPU is a
server part and the result is labelled by the registry's hardware class; the
consumer CPU points are the two Macs' own.

## 11. Reading across machines

*Added by ADR-0006 (2026-09-15), before collection.* A cell is one machine
on one device. Across machines the comparable figures are the ones §6
defines on the same cell of the same suite from the same commit: the median
and spread of inference and total per frame, the throughput, the memory
peaks - each an absolute number about that machine. The paired differences
and ratios (tensor, resolution, output mode) are computed per machine, and
what is compared across machines is whether the *ratio* holds: a tensor
effect of the same size on Metal and on CUDA is a property of the model, one
that differs is a property of the backend, and either is reported as what it
is. No figure averages machines. `unproject benchmark-campaign-summary`
folds every machine's `summary.json` into one table (`campaign.csv`) from the
raw summaries and nothing else; the thesis tables are drawn from it.
