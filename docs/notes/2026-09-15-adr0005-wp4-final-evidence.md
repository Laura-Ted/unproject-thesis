# 2026-09-15 - WP4, what a final MuPoTS result has to be, and the runbook

WP4 of ADR-0005. The tooling that makes a MuPoTS evaluation a thesis result,
and the exact procedure for regenerating the two thesis evaluations from a
clean commit. **No regeneration happened here**: the dataset and the body
artifacts are on the author's machine, not this one, and the ADR says to
document rather than pretend. What is below was tested on hand-built
evaluations and is the author's to run.

## The gate

`gt-evaluate --final` refuses, in the first second and before a model loads:

- a working tree that is dirty, or no checkout at all - the descriptor records
  a commit, and a commit has to identify the code that scored;
- `--sequence` - a subset is not the benchmark;
- and, per sequence as the bodies are opened, a body artifact whose
  `manifest.json` records a dirty tree or cannot be read - the score has to
  trace to the code that *inferred* as well, and an unknown state is not a
  clean one.

The descriptor carries `final: true` only from such a run; anything else is
`final: false` and `gt-summary` says "exploratory, not final" beside the
commit and the state of the bodies' trees. Both thesis evaluations of
2026-09-11 are `git_dirty: true` at the scoring step (ADR-0005 §2.3); the
control's bodies were inferred at `49c2dea` clean, the treatment's at
`27e0ec5` with the state not recorded in a note. `--final` will say which of
them can be rescored as they are and which must be re-inferred.

## Medians, distributions, outliers

`robust_depth` in `metrics.py`: over the signed root depth error, median,
p10/p25/p75/p90 (nearest-rank), the IQR, the MAD, the median and p90 of
|error|, the median of error over annotated distance, and two outlier counts
- beyond 1 m, a gross mis-association by any physical reading, and beyond
three scaled MADs from the median. Outliers are counted and **kept in every
mean**; the ADR forbids excluding a result for being surprising, and the
2026-09-11 note's handful of points near +4 m stay where they are. Every
evaluation now writes `summary.json` beside its CSVs: the headline numbers,
the coverage, the protocol and matcher identity, the depth distribution
overall and per bin, and the per-sequence and per-joint rows, with NaN as
null. It is the file WP7's bundle carries and the thesis tables read.

## Regeneration and comparison

`gt-summary --regenerate` rewrites `per-sequence.csv`, `per-joint.csv`,
`summary.json` and `figures/` from `observations.npz`, updates the
descriptor's references for those files only, and names any file whose bytes
changed. On a run written by this code nothing changes; on a run written by
older code the changed files are exactly the views that gained columns. The
observations are the record and are never touched.

`gt-compare a b` reads both evaluations and says whether every observation
array both carry is bit-for-bit equal (NaN equal to NaN), lists arrays one
side has and the other lacks (a schema-1 file has no scale proxy), lists
every descriptor field that differs, and writes the paired per-sequence table
and figure - depth bias, placement PCK, MPJPE, one row per sequence both
scored. Two uses: the migration check ("same numbers, new labels"), and the
camera experiment read per sequence, which the 2026-09-11 note observed no
single evaluation could draw.

## Runbook for the author, on the M3 Pro

From the release-candidate commit, with the dataset under
`datasets/mupots-3d/` and the weights in place:

```shell
git status --porcelain                     # must print nothing
uv sync --extra multihmr --extra ground-truth

# 1. rescore the existing bodies under schema 2, exploratory, to check the migration
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d.toml
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d-annotation-derived-camera.toml
uv run unproject gt-compare artifacts/ground-truth/a1a82f0677ec artifacts/ground-truth/<new control>
uv run unproject gt-compare artifacts/ground-truth/2585bc32e4d9 artifacts/ground-truth/<new treatment>
```

The treatment's bodies live under `artifacts/mupots-released-camera-TS*/`
from before the rename; step 1 for the treatment will therefore re-infer
into `artifacts/mupots-annotation-derived-camera-TS*/` (89 minutes) unless
the twenty directories are renamed first - a plain `mv` per sequence, the
artifact's identity is its hash and the name is only where it sits. Either
way, both `gt-compare` runs must print `numbers: identical` (re-inference on
the same machine has not been checked for bit-identity, per `docs/reproduce.md`
§7; a difference in the last decimal after re-inference is a re-inference
difference, not a migration one, and is to be written down as such).

```shell
# 2. the final runs: bodies from a clean tree, or re-inferred
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d.toml --final --release-frames
uv run unproject gt-evaluate experiments/ground-truth/mupots-3d-annotation-derived-camera.toml --final --release-frames
```

If `--final` refuses a body artifact as dirty or unknown, the message names
it; re-run that condition with `--overwrite` added, which re-infers every
sequence from the clean tree. Then:

```shell
# 3. read, regenerate, compare
uv run unproject gt-summary artifacts/ground-truth/<final control> --regenerate
uv run unproject gt-summary artifacts/ground-truth/<final treatment> --regenerate
uv run unproject gt-compare artifacts/ground-truth/<final control> artifacts/ground-truth/<final treatment> \
    --label-a model_estimated --label-b annotation_derived
```

Step 3's `gt-compare` writes `paired-per-sequence.csv` and
`paired-depth-bias.svg`, the per-sequence camera figure. The ids of the two
final evaluations, the `numbers: identical` lines from step 1, and anything
`--final` refused go into a dated note; WP7's bundle then carries the two
`summary.json` files, the CSVs and the paired outputs.

What the thesis may say afterwards follows ADR-0005 §3 and the protocol's
§7a; nothing in this note changes a number.

## Verified here

24 new tests: the robust statistics (nearest-rank, outliers under both
definitions, empty input, shape mismatch); the summary (names what it is,
distribution beside the mean, NaN as null and back, per-sequence rows,
deterministic); regeneration reproducing a run byte for byte and naming a
stale view; the final gate (dirty, no checkout, subset, dirty or unknown
bodies, refused before the dataset is touched); comparison (identical, a
moved number named with its size, schema 1 against 2, descriptor fields,
deterministic paired outputs, different skeletons refused). One CLI test runs
`gt-summary --regenerate` and `gt-compare` end to end on a synthetic
evaluation.
