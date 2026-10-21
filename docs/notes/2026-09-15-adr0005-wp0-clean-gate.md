# 2026-09-15 - WP0, a gate that passes from a clean checkout

WP0 of ADR-0005. The public CI of `03537a9` through `fbc810f` was red in both
jobs, and the local suite was green only because the development `.venv` had
accumulated packages the lockfile does not install by default. Three defects,
two of them named by the ADR and one it could not have known about.

## What was wrong

**Prettier.** `apps/web/src/App.tsx` carried two blank lines where one is
allowed. `pnpm format` removed one line. Nothing else.

**The ground-truth extra.** The pipeline job ran `uv sync --dev`. The tests for
the MuPoTS adapter and the global matcher import `scipy`, which only the
`ground-truth` extra installs, so on a clean checkout 22 of them failed with
`ModuleNotFoundError`. Measured, not assumed: an environment built with
`uv sync --dev` and nothing else fails exactly 22 and passes 584. The job now
installs `--extra ground-truth`, its comment no longer claims that no extra is
needed, and the README and `docs/reproduce.md` name the extra in the gate and
in the MuPoTS commands. Moving SciPy into the base dependencies was the
ADR's alternative and was not taken: nobody running the viewer needs a `.mat`
reader.

**A newly tracked script.** `fbc810f` committed `docs/thesis-figures/`, which
the ADR had described as untracked local content. `generate.py` had never met
the linter: 73 ruff errors and one file to reformat, so `ruff check .` failed
in CI from that commit on. It is now formatted, typed, and lint-clean, with
the same behaviour: four figures from the same three sources, the same
`provenienza.json`. The typographic `×` and `−` in axis labels are kept and
marked deliberate rather than replaced. It is not deleted and not moved: the
author versioned it, and WP7 decides what reads the evidence bundle.

## What the clean environment found that the ADR did not

The ADR asks that the suite run in an environment rebuilt from the lockfile.
Doing that - `UV_PROJECT_ENVIRONMENT=/tmp/unproject-clean uv sync --dev
--extra ground-truth` - the main pytest invocation died with SIGABRT at
`test_pose_stage.py`, exactly the libomp collision the 2026-09-10 note fixed
for `test_pose_colmap.py`. `run_pose` opens the COLMAP backend, and with it
pycolmap, in the same process where `test_benchmark.py` had already imported
torch.

It passed in the development `.venv` by luck: with `multihmr2`, `anny` and
`sklearn` present, the load order happened not to trip the duplicate-runtime
check, and the same two modules imported in a bare interpreter abort in both
environments. A gate that depends on what else is installed is not a gate.

The split is now enforced rather than remembered:

- the tests that reach pycolmap live under
  `packages/pipeline/tests/needs_pycolmap/` - the two modules, `test_pose_colmap`
  and `test_pose_stage`;
- `packages/pipeline/tests/openmp_guard.py`, a pytest plugin loaded from
  `pyproject.toml`, leaves that directory out of a plain `uv run pytest` and
  collects only it under `uv run pytest --pycolmap`. The decision is made in
  `pytest_ignore_collect`, before any module is imported, because a marker on
  its own is not enough: pytest imports a deselected module to read its
  markers, and `import pycolmap` at the top of one is already the collision;
- a test outside the directory that imports pycolmap, or one inside it that
  imports torch, fails with the two commands in its message; a selection that
  mixes the two groups is refused at collection time.

The CI, README and reproduction guide use the two new invocations. The
2026-09-10 note has a superseding preamble.

## Verification

From `/tmp/unproject-clean`, built from the lockfile with `--dev --extra
ground-truth` and without `multihmr`:

```
uv run ruff check .                     All checks passed
uv run ruff format --check .            141 files already formatted
uv run mypy                             no issues in 103 source files
uv run pytest                           606 passed
uv run pytest --pycolmap                18 passed
uv run unproject validate-config experiments/   30 ok
uv lock --check                         ok
pnpm install --frozen-lockfile; format:check; compile; lint; test (97); build; audit --prod   all ok
```

624 tests, the same count as before: three moved from the main invocation to
the isolated one. Without the extra: 22 failed, 584 passed, which is the number
the ADR predicted.

## Not done here

The remote CI must be seen green on the pushed commit (WP0 step 5). That is
observed after the push, not asserted before it.
