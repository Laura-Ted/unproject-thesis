# Reproducing this

From a clean checkout, in order. Every step below was run from a fresh clone on
2026-09-06 before this file was written; where a step needs something the
repository cannot carry, it says so.

## What you need

| | |
|---|---|
| Machine | Apple Silicon on macOS (Metal), or Linux x86-64 with an NVIDIA card whose driver supports CUDA 13 (ADR-0006). Anything else runs the model on the CPU, slowly, and says so. |
| Python | Installed by `uv` itself, pinned to 3.12 |
| Node | For the viewer, via `pnpm` |
| Disk | ~3 GB of wheels for the estimator on macOS, ~6 GB on Linux (the CUDA libraries come with torch), plus 1.6 GB of weights on first use |

The checks in CI run on Linux and need neither of the last two. The device is
chosen as CUDA, then Metal, then CPU; `UNPROJECT_DEVICE=cpu` (or `cuda`,
`mps`) insists on one, and a device that is asked for and absent is an error
rather than a fallback.

## 1. Install and start it

```shell
uv sync --extra multihmr    # without --extra you get everything but inference
pnpm install
uv run unproject-api
```

Then open **http://localhost:8000**. That one command builds the viewer if the
bundle is older than its sources and serves it; there is no separate build step.

With no artifacts on disk the run list is empty, which is the correct empty
state. Press **Open a video** or **Use the camera** to start from nothing.

**Without `--extra multihmr`** the API still starts, still serves finished runs,
and the viewer still draws them. Only the two buttons above refuse, and they say
which extra to install rather than failing generically.

## 2. Fetch a scene

The repository carries no footage. One public-domain clip is fully described by
its config, so it can be fetched and will hash to the same bytes:

```shell
mkdir -p datasets/_download datasets/pedestrians
curl -L -o datasets/_download/pedestrian_area.webm \
  https://upload.wikimedia.org/wikipedia/commons/a/ae/Video_Codec_Test_pedestrian_area_1080p25.y4m.webm
shasum -a 256 datasets/_download/pedestrian_area.webm
# expect bfadaa62cccb42db875d50bb842aa0964fbf72040432e4097c1df59e043e0c26
ffmpeg -y -i datasets/_download/pedestrian_area.webm \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an datasets/pedestrians/clip.mp4
```

CC0 1.0, so nothing about using it or publishing a frame of it is in question.
`experiments/pedestrians.toml` repeats all of the above in its own comments.

The other scene in `experiments/`, `crowd`, is **not** reproducible: its footage
is not public and its provenance was never recorded. See §8 of the protocol.

## 3. Fit bodies to it

```shell
uv run unproject bodies experiments/pedestrians.toml
```

About four minutes for 375 frames on an M3 Pro, plus roughly ten seconds to
build the model the first time in a process. It writes to
`artifacts/pedestrians/bodies/<hash>/`. Re-running the same config over the same
bytes reuses what is there; the hash changes if either changes.

Reload the viewer and the run appears in the list.

## 4. Measure it

Protocol v2, the final RQ1/RQ2 evidence (`docs/experiments/protocol-v2.md`),
runs its matrix on MOT16-09 (`benchmarks/fetch-datasets.sh` fetches the
official archive, §4b) at four geometries the ingest makes from the same
JPEG frames - `experiments/mot16-09.toml`, `mot16-09-half.toml`
(`[ingest.geometry] downscale = 2`), `mot16-09-letterbox.toml`
(`letterbox_square = true`) and `mot16-09-letterbox-half.toml` (both) - so
every machine derives the same bytes from the same archive and nothing is
transcoded. The pedestrians clip and the ffmpeg transformations below are
what the pilot of 2026-09-15 morning used, kept as its record:

```shell
ffmpeg -y -i datasets/pedestrians/clip.mp4 -vf "scale=960:540" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an datasets/pedestrians-half/clip.mp4
ffmpeg -y -i datasets/pedestrians/clip.mp4 -vf "pad=1920:1920:0:420:color=black" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an datasets/pedestrians-letterbox/clip.mp4
ffmpeg -y -i datasets/pedestrians/clip.mp4 -vf "pad=1920:1920:0:420:color=black,scale=960:960" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an datasets/pedestrians-letterbox-half/clip.mp4
```

The campaign's matrix:

```shell
uv run unproject benchmark-suite experiments/benchmark/v2/matrix-suite.toml
uv run unproject benchmark-suite-summary experiments/benchmark/v2/matrix-suite.toml
```

Or, as the campaign runs it on every machine - preflight, suite and summary,
under a root of the machine's own:

```shell
make bench-preflight MACHINE=B        # ids: benchmarks/machines.toml
make bench-matrix MACHINE=B           # also bench-people, bench-cpu, bench-all
```

Twenty-four jobs, each a fresh process that loads the model, so allow the
model load - about ten seconds with the weights in the page cache - twenty-four
times over on top of the frames. On mains, low-power mode off, nothing else
running: the job records the power source and the thermal CPU limit before and
after, and the summary reports them. Every job result lands under
`artifacts/benchmark-v2/rq1rq2-v2-matrix/<cell>/`, and the summary writes
`summary.json`, `summary.csv` and `pairs.csv` beside them without running
inference.

### 4b. The person-count factor and the CPU point

Two more suites under the same protocol (ADR-0006; protocol v2 §4.2 and §10).
The people suite needs MOT16-08 and MOT16-09 from MOTChallenge and
MuPoTS-3D TS10 and TS11 from MPI for Informatics, all read as the JPEG frames
they ship as;
`benchmarks/fetch-datasets.sh` fetches and verifies them (`docs/datasets.md`);
the MuPoTS terms are yours to accept, and the script fetches those two
sequences only when told `--accept-mupots-terms`. The CPU suite holds the
model to the CPU by its own `device = "cpu"` line.

```shell
benchmarks/fetch-datasets.sh --accept-mupots-terms
uv run unproject benchmark-suite experiments/benchmark/v2/people-suite.toml --machine B --artifacts-root artifacts/campaign/B
uv run unproject benchmark-suite experiments/benchmark/v2/cpu-suite.toml --machine B --artifacts-root artifacts/campaign/B
uv run unproject benchmark-campaign-summary artifacts/campaign
```

The last folds every machine's summaries under `artifacts/campaign/<id>/`
into `campaign.csv`, from the summaries and nothing else. The campaign has
run: seven machines, `docs/notes/2026-09-16-campaign-report.md` says where
the results are and how to validate them (`benchmarks/validate-campaign.py`).

The v1 benchmark below is the pilot; it is kept runnable but produces no thesis
number.

```shell
uv run unproject benchmark experiments/benchmark/crowd-skeleton.toml
uv run unproject benchmark-summary
```

### 4a. Measure the live path

`docs/experiments/live-protocol.md`. The transport client replays the clip
into the real socket at a controlled rate and needs no browser; `--spawn`
starts the server on the port in `--url` and stops it afterwards:

```shell
uv run unproject-live-benchmark datasets/pedestrians/clip.mp4 --fps 1 --spawn
uv run unproject-live-benchmark datasets/pedestrians/clip.mp4 --fps 8 --spawn
uv run unproject-live-benchmark datasets/pedestrians/clip.mp4 --fps 1 --bodies --spawn
```

The browser session adds the render side. Start the server, open the URL,
and wait for the status bar to say where the report was kept:

```shell
uv run unproject-api
open "http://localhost:8000/?measure=1&replay=datasets/pedestrians/clip.mp4&fps=1"
open "http://localhost:8000/?measure=1&replay=datasets/pedestrians/clip.mp4&fps=8"
```

Then `uv run unproject live-summary` writes `artifacts/live/summary.csv` and
prints one line per report, transport and browser kept apart.

Each `benchmark` invocation writes one raw result, named by when it started, and
never replaces an earlier one. `benchmark-summary` folds every result into
`summary.json` and `summary.csv` and runs no inference, so the table regenerates
on a machine with no weights.

`--dry-run` exercises the whole path with a stand-in estimator, measures nothing,
says so, and is excluded from the summary.

The matrix that produced the numbers in the README is the eight configs in
`experiments/benchmark/crowd*.toml`, and it needs the crowd footage. Point them
at `pedestrians.toml` instead to run the same shape of experiment on footage you
can actually fetch — the numbers will differ and the protocol's §2.1 confounder
note will not apply.

## 5. Score the stability

```shell
uv run unproject rq3 artifacts/pedestrians/bodies/<hash> \
    experiments/rq3/pedestrians-uncontrolled.toml
```

Writes `rq3.json`, `rq3-positions.csv` and the figures, runs no inference, and is
deterministic: run it twice and the bytes match.

That annotation file declares **no floor marks**, so no position error is
computed and none is reported. Measuring how far the bodies are from the truth
needs marks measured with a tape and a consenting participant, which is a
recording nobody has made. `experiments/rq3/corridor.example.toml` is the
template for when someone does.

### 5a. Characterise the filter

```shell
uv run unproject filter-characterise
```

Deterministic and instant: the viewer's root filter over synthetic noise,
steps, ramps and sinusoids with a known truth, at the cadences and distances
the viewer sees (`docs/experiments/filter-characterisation.md`). It writes
`artifacts/filter/` - a JSON report, a CSV and two figures - and reports
smoothness and delay, never accuracy.

## 6. Run the checks

```shell
uv sync --dev --extra ground-truth
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv run pytest --pycolmap
uv run unproject validate-config experiments/
uv lock --check

pnpm install --frozen-lockfile
pnpm format:check && pnpm compile && pnpm lint && pnpm test && pnpm build
pnpm audit --prod
```

None of these need the weights or any footage; the tests synthesise the video
they need. This is what `.github/workflows/ci.yml` runs, and
`uv run unproject gates` runs the same list as one command - the list is
`unproject.gates.GATES`, held equal to the workflow by a test - without
stopping at the first failure, and writes `artifacts/gates.json` with the
exit code, the duration and the output tail of each. That record is what the
evidence bundle carries (§8). On 2026-09-15 it was run from a fresh
`git clone` on macOS with nothing installed but `git`, `uv` and `pnpm`, and
came back green; it needs no Apple Silicon.

The one extra is not optional. `ground-truth` installs the reader for MuPoTS's
`.mat` files, and the tests that cover the adapter and the global matcher
import it; on a checkout synced without it, 22 of them fail with a
`ModuleNotFoundError` instead of skipping. That is what turned CI red between
2026-09-10 and 2026-09-14, and it was invisible locally because the development
environment already had the package (ADR-0005 §2.2). Verify the gate from an
environment rebuilt from the lockfile, not from a `.venv` that has accumulated
packages: `UV_PROJECT_ENVIRONMENT=/tmp/unproject-clean uv sync --dev --extra
ground-truth` builds one beside the repository without touching `.venv`.

`pytest` is two invocations, not one, and on macOS it has to be. `pycolmap` and
`torch` each ship a copy of libomp, and a process that loads both aborts - not
a test failure, a dead process, after which pytest prints nothing and the exit
code is the only evidence. On Linux the two happen to coexist. See
`docs/notes/2026-09-10-the-suite-was-aborting-at-test-24.md`.

The split is enforced rather than remembered. Every test that reaches
`pycolmap` lives under `packages/pipeline/tests/needs_pycolmap/`, and the
plugin `packages/pipeline/tests/openmp_guard.py` - loaded by `pyproject.toml`
- leaves that directory out of a plain `pytest` and collects only it under
`--pycolmap`, before any module is imported. A test elsewhere that imports
`pycolmap`, or one in there that imports `torch`, fails naming the two
commands. Until 2026-09-15 the split was by file name and missed
`test_pose_stage.py`, which opens the COLMAP backend: the suite survived on
the development machine by load order alone and died on a checkout rebuilt
from the lockfile.

## 7. Evaluate against a public ground truth

This is the final experiment of the thesis and the only place its depth claim
is measured against a truth someone else recorded. It needs a dataset this
repository does not contain and must not contain.

### Obtaining MuPoTS-3D

The licence (MPI for Informatics, 2018) permits non-commercial research only,
forbids redistribution, and requires two citations - Mehta et al. 3DV 2018 and
Mehta et al. 3DV 2017 - in any resulting publication. Reading and accepting it
is the author's act, not the tooling's: nothing here downloads it, and the
official download script ships with its download lines commented out for the
same reason.

Go to the project page named in `experiments/ground-truth/mupots-3d.toml`,
read the licence, obtain `MultiPersonTestSet.zip` (about 5.6 GB), and extract
it so that this exists:

```text
datasets/mupots-3d/MultiPersonTestSet/TS1/annot.mat
datasets/mupots-3d/MultiPersonTestSet/TS1/img_000000.jpg
…
datasets/mupots-3d/MultiPersonTestSet/TS20/
```

Twenty `TS` directories, each with `annot.mat`, `occlusion.mat` and numbered
JPEG frames. `datasets/` is ignored by git. Do not obtain the archive from
anywhere but the official page: a mirror could differ, and the evaluation
records a content hash of the annotation files that a differing copy would not
match.

The evaluation refuses to start if any of the twenty sequences is missing,
naming which. An evaluation over fewer is possible with `--sequence`, is given
a different identity, and is not the benchmark.

### What it costs

| | |
|---|---|
| Disk, intermediate | **27.9 GB** of decoded frames. Stage [1] writes every frame as lossless PNG, about five times the JPEG. Pass `--release-frames` to drop each sequence's frames once its bodies are written; peak becomes one sequence, 2.5 GB. Nothing measured is lost and the frames regenerate. |
| Disk, kept | about 0.5 GB of body artifacts, plus a few MB per evaluation |
| Time | **89 minutes** of inference on an M3 Pro for the twenty sequences, 0.72 s per frame. Once per camera condition; rescoring is seconds. |
| Weights | the same pinned checkpoint as every other stage; fetched on first use |

### Running it

```shell
uv sync --extra multihmr --extra ground-truth
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d.toml
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d-annotation-derived-camera.toml
```

The first is the control - the model guesses its camera
(`camera = "model_estimated"`). The second is the treatment - it is told each
sequence's intrinsics, recovered by least squares from the released annotations
(`camera = "annotation_derived"`; MuPoTS publishes no calibration). The two
files differ in that one line, and their comments say so. Both declare the
metric protocol, `unproject_mupots_14j_v1`, which is the project's
MuPoTS-compatible metric and not the official evaluator's score. Each prints a line per sequence as it goes and the headline numbers at the
end, and writes `artifacts/ground-truth/<id>/` where `<id>` is a hash of the
configuration, the annotation files, the joint mapping and the sequence list.
Two runs of the same thing land in the same directory and the second refuses
to overwrite without `--overwrite`; two different things never share one.

Body artifacts are reused across evaluations: the treatment run re-infers
because its bodies are different, but rescoring either condition after a change
to the metrics runs no inference at all.

### Making it final

A thesis result is a run with `--final` (ADR-0005 WP4):

```shell
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d.toml --final --release-frames
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d-annotation-derived-camera.toml --final --release-frames
```

It refuses, before loading anything, a working tree that is dirty or a subset
of sequences, and it refuses to score bodies whose own `manifest.json` records
a dirty tree or none at all - so every number traces to a commit that
identifies the code that inferred *and* the code that scored. Bodies produced
earlier from a dirty tree have to be re-inferred (`--overwrite`), which is the
89 minutes per condition above. The descriptor records `final: true` and the
commit and state of every body artifact; `gt-summary` prints them. A run
without `--final` is an exploratory run and its descriptor says so.

### Reading it

```shell
uv run unproject gt-summary artifacts/ground-truth/<id>
uv run unproject gt-summary artifacts/ground-truth/<id> --regenerate
```

No weights, no dataset, no inference: it reads `evaluation.json` and
`observations.npz` and prints what `gt-evaluate` printed, plus the depth error
as a distribution - median, quartiles, p10/p90, the median of |error|, the
relative median - and the outliers counted under two definitions (beyond 1 m,
beyond 3 scaled MADs), which stay in every mean. The directory also holds
`per-sequence.csv`, `per-joint.csv`, `summary.json` and `figures/`, all
derived from `observations.npz`. `--regenerate` rewrites them from the
observations and says which, if any, changed: a table in the thesis is drawn
from files that a reader can reproduce with that one command.

### Testing the hypotheses

```shell
uv run unproject gt-hypotheses artifacts/ground-truth/<final treatment>
uv run unproject gt-hypotheses artifacts/ground-truth/<final control>
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d-letterbox-square.toml --final --release-frames
uv run unproject gt-compare artifacts/ground-truth/<final treatment> artifacts/ground-truth/<letterbox> \
    --label-a wide --label-b letterboxed
```

The first two run the pre-registered body-scale test of the protocol's §4.6 on
evaluations that carry the scale proxy (schema 2) and write
`hypotheses.json`, a per-sequence CSV and a figure under
`<evaluation>/hypotheses/`, with the reading rule in the file and the post-hoc
diagnostics labelled. The third is the controlled square-frame test - the
fifteen wide sequences letterboxed to squares, about an hour of inference,
padded frames under `artifacts/_derived/` released per sequence - and the
fourth reads it against the plain run, per sequence.

### Comparing two

```shell
uv run unproject gt-compare artifacts/ground-truth/<old> artifacts/ground-truth/<new>
uv run unproject gt-compare artifacts/ground-truth/<control> artifacts/ground-truth/<treatment> \
    --label-a model_estimated --label-b annotation_derived
```

The first use is the migration check: the same bodies rescored under schema 2
must give `numbers: identical` - every observation array bit-for-bit equal -
with only labels and identifiers in the descriptor diff. The second is the
camera experiment read per sequence: `paired-per-sequence.csv` and
`paired-depth-bias.svg`, the figure no single evaluation can draw, from the
two observation files and nothing else.

`observations.npz` is the source of truth: one row per matched person, one per
frame, no aggregates, no object arrays. A question nobody asked before the run
can be answered from it after.

### What to expect

Rescoring existing body artifacts reproduces the README's numbers to the
decimal; that has been checked by regenerating both evaluations. Evaluations
written before 2026-09-15 are schema 1 and read back under the schema-2 names
- `gt-summary` says so in a `note:` line - and rescoring the same bodies under
schema 2 lands under a new id with the same numbers. Whether a
fresh inference run on another Apple Silicon machine produces bit-identical
bodies has **not** been checked - no second machine was available - so treat
small differences in the third significant figure as unexplained rather than
as wrong. The descriptor records the annotation inventory (`25e98ea86b61…`)
and the checkpoint (`fd2d9ab7010a…`) so that a differing input is at least
ruled out first.

### What is still not measured

- **MPJVE**, the velocity error. The release states no frame rate and the
  config declares 30 fps as a placeholder it says is unsourced. Nothing else
  depends on it.
- **Precision.** MuPoTS annotates a subset of the people in frame, so an
  unmatched prediction is not evidence of anything. Recall only.
- **Anny-One.** ADR-0004 WP5-WP6. Not started; the ADR itself names the MuPoTS
  result as the minimum complete benchmark.

## 8. Build the evidence bundle

Every number the thesis quotes is carried, with its hash, by a bundle the
repository versions; the artifacts it was drawn from are not. After the final
runs of §4, §4a, §5a and §7, from a clean commit:

```shell
uv run unproject gates
uv run unproject evidence-bundle experiments/evidence/thesis.toml evidence/thesis \
    --gates artifacts/gates.json
uv run unproject evidence-verify evidence/thesis
```

`experiments/evidence/thesis.toml` names each result and the command that
produced it; until the final runs exist every result is optional and the
bundle records it as absent. The builder refuses an evaluation that is not
final, a suite with a job from a dirty tree, a run whose manifest does not
record a clean tree, and any file over 2 MB; `observations.npz`, the bodies
fitted to dataset frames, footage and weights are indexed by SHA-256 and stay
local. `evidence-verify` re-hashes a checkout's bundle against `release.json`.
The thesis's tables and figures are regenerated by the scripts of the thesis
repository from stored results, with a manifest naming the file and hash
behind each. The order of everything from the final runs to the tag is
`docs/release.md`.

## What you cannot reproduce from here

- **The crowd clip.** Not public, provenance unrecorded. Every number in the
  README drawn from it is labelled with the run that produced it, but you cannot
  regenerate that run.
- **The RQ3 controlled case study.** It does not exist and was superseded by
  §7. No position error is claimed from it anywhere.
- **MuPoTS-3D itself.** Not redistributed, for the reasons in §7; with the
  licensed files in place, everything computed from them reproduces.
- **Nothing of the campaign, given the inputs.** The three suites ran on
  seven machines on 2026-09-15/16 from one commit; every job, summary and
  the cross-machine table are under `artifacts/campaign/` on the operator's
  machine and in the thesis repository, and regenerate from the raw jobs
  with `make bench-summary` / `make bench-campaign` without inference
  (`docs/notes/2026-09-16-campaign-report.md`). Re-measuring is a matter of
  a machine and an hour, not of code.
- **The old live transport figure** the README once quoted. It was withdrawn;
  its replacement is collected under `docs/experiments/live-protocol.md` (§4a
  above) and is reproducible.
