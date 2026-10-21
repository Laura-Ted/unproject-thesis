# 2026-09-23 - Code cleanup, and an erratum to protocol v2 §6

A readability pass over the code after the campaign, with one rule: nothing the
thesis quotes may change. It touched comments, a little code that was never
called, and files that nothing uses any more. The measurement and evaluation
code computes exactly what it computed before.

## What changed

**Comments** (`50e3db4`). About seventy comments and docstrings no longer said
what the code does, and were corrected; duplicated rationale and anecdotes were
cut. Among the corrections: the v2 summary's median (below); "ADR-0002 §4.2",
cited for the libomp rule, does not exist - the rule is in `docs/reproduce.md`
§6 (the note of 2026-09-10 still cites it and is left as written); three points
do not fit a similarity exactly; the joint mapping's package version and
checkpoint are recorded, not compared; the CI split is made by the
`openmp_guard` plugin, not a root conftest.

**Two small refactors** (`0c5a2c5`): a `torch` import that `_collect_frame`
never used, and one helper for the eleven identical `--artifacts-root` options.

**Code that nothing called:** `Estimator.worker`, `ground_truth.figures.summary_line`,
`pose.geometry.opengl_to_opencv`, `MupotsDataset.missing_sequences`,
`Aggregate.is_empty`, `Distribution.iqr`; in the viewer `duration`, `advance`,
`PlaybackClock`, the `MeshAssets` type and three CSS rules no element uses.
Tests that exercised only the removed code went with it.

**Files nothing uses:** `package-lock.json` (npm; `pnpm-lock.yaml` is the
lockfile), `benchmarks/add-recording.py` (for the author's clip, which the
campaign dropped on 2026-09-15), `docs/thesis-figures/` and
`docs/thesis-preview/`. The last two were removed with the author's agreement,
which ADR-0005 asks for: the thesis draws its tables and figures with the
scripts of its own repository, and nothing read these folders.

**A flaky test.** `App.test.tsx` "asks before it deletes anything" clicked a
row before the run list had rendered; under load it failed 2 times in 30. It
now waits for the list, as the tests beside it do.

## How it was checked

Before any change, the reference outputs were recorded on the Mac mini at
`94c084a`; after each group of changes they were compared:

- comments: the AST of every changed Python file is identical once docstrings
  are removed, every code line is identical character for character, and the
  TypeScript token stream without comments is identical;
- removals: every remaining statement is identical; only unreferenced
  definitions are gone;
- the gates, the viewer's JavaScript bundle (byte-identical), the `--help` of
  every command and the OpenAPI schema;
- the campaign summaries regenerated from the raw jobs, byte-identical to the
  ones the thesis copied, and the thesis's own `--check` for the campaign and
  the filter;
- for the comments and the two refactors, a re-run of ingest, benchmark jobs,
  bodies, rq3 and a MuPoTS-3D subset: 6376 files, identical apart from time,
  memory and `git_dirty`. A bodies run was also byte-identical to one made on
  2026-09-15 at `c5c72be`: inference is deterministic on this machine.

`rq3/metrics.py`, `rq3/reader.py`, `rq3/smoothing.py` and
`benchmarks/machines.toml` were not touched: the thesis records their SHA-256.

## Erratum to protocol v2 §6

§6 says "All percentiles are nearest-rank, so a reported value is one that was
measured", which reads as if the median were one of them. It is not:
`unproject.benchmark.summary` reports the middle value, or the mean of the two
middle values for an even count, and every campaign cell has 3 × 60 = 180
samples. The thesis tables compute the median the same way (`numpy.median`),
so code and thesis agree. p25, p75, p90 and p95 are nearest-rank, as stated.

Every result of the campaign was collected and summarised with this definition,
so nothing is recomputed and no number changes; only the wording is corrected.
For scale: a nearest-rank median would move the 126 campaign medians by 0.018 ms
at the median and 2.9 ms at most (0.15 %).
