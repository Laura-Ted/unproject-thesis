# 2026-09-10 - The pipeline suite was aborting at test 24, and reporting nothing

> **Superseded on 2026-09-15 (ADR-0005 WP0).** The split by file name below
> missed `test_pose_stage.py`, which reaches pycolmap through the COLMAP backend,
> and a checkout rebuilt from the lockfile aborted again. The two invocations
> are now `uv run pytest` and `uv run pytest --pycolmap`, decided at collection
> time by `packages/pipeline/tests/openmp_guard.py`, which also fails any test
> that crosses the boundary. The mechanism this note said was missing exists.

Running the project's sanity check on this machine:

```
$ uv run pytest packages/pipeline apps/api
........................
$ echo $?
134
```

Twenty-four dots, no summary line, no failure, exit 134 — SIGABRT. Under `-s`
the reason appears:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
initialized.
```

There are three copies of libomp in the virtualenv:

| package  | sha256 (first 12) |
|----------|-------------------|
| torch    | `6256bee09e93` |
| sklearn  | `7fe04d06d824` |
| pycolmap | `5d67ec212fbc` |

pycolmap is imported at collection time by `test_pose_colmap.py`; torch is
imported by the first test in `test_benchmark.py::TestOrchestration`, and that
is test 25. Whichever loads second aborts — the order does not matter, both
directions were tried:

```
$ uv run python -c "import torch, pycolmap"     # aborts
$ uv run python -c "import pycolmap, torch"     # aborts
```

## This was already known, in one place only

`apps/api/src/unproject_api/app.py` opens with "This process must never import
pycolmap… the two abort each other; see ADR-0002 §4.2". The API process obeys
that rule. The test suite did not, and the failure mode is the bad kind: the
process dies mid-run, so pytest never prints a summary and the exit code is the
only evidence anything went wrong. 491 of 515 tests were not running.

CI stayed green throughout, because on Linux torch links libgomp and the
collision does not occur. A gate that passes on the runner and aborts on the
author's machine is not a gate.

## What was changed

Nothing about the tests themselves, and no `KMP_DUPLICATE_LIB_OK` — the OpenMP
documentation calls that workaround unsafe and undocumented, and it would trade
a loud abort for possible silent wrong answers.

The single pytest invocation became two, in `.github/workflows/ci.yml`, giving
the pycolmap module a process of its own:

```shell
uv run pytest --ignore=packages/pipeline/tests/test_pose_colmap.py
uv run pytest packages/pipeline/tests/test_pose_colmap.py
```

Both are green: 500 and 15. Run locally in that order rather than as one
command; the project checklist still documents the single form.

## What this does not fix

Any future test module that imports both sides in one process will abort the
same way. The constraint lives in ADR-0002 §4.2 and in a docstring; there is no
mechanism enforcing it. Worth an ADR if a third such library ever appears.
