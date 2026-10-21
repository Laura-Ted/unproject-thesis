"""A pytest plugin with one rule.

Loaded through `-p openmp_guard` in `pyproject.toml` rather than as a conftest:
`pytest_addoption` has to run at startup, the two test roots have no common
conftest directory, and a plugin is an ordinary module that mypy and ruff
check like any other.

The rule, and it is a process-boundary rule rather than a test rule: **pycolmap
and torch must never be imported into the same process.** Each bundles its own
copy of libomp, and whichever initialises second aborts the interpreter with
SIGABRT - no traceback, no summary, and an exit code as the only evidence. On
Linux the two happen to coexist, which is how CI stayed green while the suite
was dying on the machine the project targets (`docs/reproduce.md` §6,
`docs/notes/2026-09-10-the-suite-was-aborting-at-test-24.md`).

Whether the abort fires also depends on what else was loaded first: one
development environment survived the full suite by load order alone, and a
checkout rebuilt from the lockfile died at the first collision. A gate that
depends on luck is not a gate, so the constraint is enforced here rather than
remembered.

Tests that reach pycolmap live under `packages/pipeline/tests/needs_pycolmap/`,
and the split is made at collection time, before any module is imported - a
marker alone is not enough, because pytest imports a deselected module to read
its markers, and `import pycolmap` at the top of one is already the collision:

    uv run pytest              # everything except that directory
    uv run pytest --pycolmap   # only that directory, in a process of its own

A test outside the directory that imports pycolmap, or one inside it that
imports torch, fails with those two commands in the message rather than
passing by accident.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

PYCOLMAP_MARKER = "pycolmap"
PYCOLMAP_DIRECTORY = Path(__file__).parent / "needs_pycolmap"

_TWO_COMMANDS = (
    "Run the two groups in separate processes:\n"
    "    uv run pytest              # everything else\n"
    "    uv run pytest --pycolmap   # packages/pipeline/tests/needs_pycolmap only"
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--pycolmap",
        action="store_true",
        default=False,
        help="collect only the tests that import pycolmap; they cannot share a process "
        "with the rest of the suite",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Decide the group by directory, before anything under it is imported."""
    inside = collection_path.resolve().is_relative_to(PYCOLMAP_DIRECTORY.resolve())
    wanted = bool(config.getoption("--pycolmap"))
    if inside and not wanted:
        # The whole directory, so nothing in it is even imported.
        return True
    if wanted and not inside and collection_path.is_file():
        # Files only: the directories above the pycolmap one still have to be
        # walked to reach it.
        return True
    return None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark the isolated group, then check that nothing else was collected with it."""
    for item in items:
        if item.path.resolve().is_relative_to(PYCOLMAP_DIRECTORY.resolve()):
            item.add_marker(pytest.mark.pycolmap)
    marked = [item for item in items if item.get_closest_marker(PYCOLMAP_MARKER) is not None]
    if marked and len(marked) != len(items):
        unmarked = next(item for item in items if item.get_closest_marker(PYCOLMAP_MARKER) is None)
        raise pytest.UsageError(
            f"{marked[0].nodeid} imports pycolmap and {unmarked.nodeid} does not; they were "
            f"collected into one process, which aborts once both libomp copies load. "
            f"{_TWO_COMMANDS}"
        )
    if marked and "torch" in sys.modules:
        raise pytest.UsageError(
            "this is the pycolmap invocation, and collecting it already imported torch; a "
            f"module in that directory must not import torch at module level. {_TWO_COMMANDS}"
        )


@pytest.fixture(autouse=True)
def _one_openmp_runtime_per_process(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail the test that crosses the boundary, even when the abort does not fire.

    Only the test that caused the import is blamed: the module stays loaded for
    the rest of the session, and failing every later test would hide the culprit.
    """
    marked = request.node.get_closest_marker(PYCOLMAP_MARKER) is not None
    forbidden = "torch" if marked else "pycolmap"
    loaded_before = forbidden in sys.modules
    yield
    if loaded_before or forbidden not in sys.modules:
        return
    if marked:
        pytest.fail(
            f"{request.node.nodeid} imported torch from inside the pycolmap group. The two "
            f"cannot share a process. {_TWO_COMMANDS}"
        )
    pytest.fail(
        f"{request.node.nodeid} imported pycolmap. Move its module under "
        f"packages/pipeline/tests/{PYCOLMAP_DIRECTORY.name}/ so it runs in the isolated "
        f"invocation. {_TWO_COMMANDS}"
    )
