"""The quality gates, as one list, run and recorded.

ADR-0005 WP7 asks the evidence bundle to carry the state of the automated
gates. The list lives here and nowhere else: the CI workflow runs the same
commands, and a test parses `.github/workflows/ci.yml` and fails if the two
ever disagree, so "green" means the same thing on the runner and in a bundle.

`unproject gates` runs every command from the repository root, records the
exit code, the duration and the tail of the output, and writes `gates.json`.
It does not stop at the first failure: a bundle should say which gates
failed, not only that one did.
"""

from __future__ import annotations

import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

GATES_SCHEMA_VERSION = 1
GATES_FILENAME = "gates.json"

# In the order CI runs them. Each is one shell-free argv.
PYTHON_GATES: tuple[tuple[str, ...], ...] = (
    ("uv", "sync", "--dev", "--extra", "ground-truth"),
    ("uv", "run", "ruff", "check", "."),
    ("uv", "run", "ruff", "format", "--check", "."),
    ("uv", "run", "mypy"),
    ("uv", "run", "pytest"),
    ("uv", "run", "pytest", "--pycolmap"),
    ("uv", "run", "unproject", "validate-config", "experiments/"),
    ("uv", "lock", "--check"),
)
VIEWER_GATES: tuple[tuple[str, ...], ...] = (
    ("pnpm", "install", "--frozen-lockfile"),
    ("pnpm", "format:check"),
    ("pnpm", "compile"),
    ("pnpm", "lint"),
    ("pnpm", "test"),
    ("pnpm", "build"),
    ("pnpm", "audit", "--prod"),
)
GATES: tuple[tuple[str, ...], ...] = (*PYTHON_GATES, *VIEWER_GATES)

# How much of a gate's output is kept. Enough to read a failure, not a log.
_TAIL_CHARACTERS = 4000


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str
    exit_code: int
    seconds: float = Field(ge=0.0)
    output_tail: str


class GatesRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = GATES_SCHEMA_VERSION
    started_at: datetime
    finished_at: datetime
    git_commit: str | None
    git_dirty: bool
    # True only when every gate exited zero.
    green: bool
    results: list[GateResult]


def run_gates(root: Path, *, commands: tuple[tuple[str, ...], ...] = GATES) -> GatesRecord:
    """Run every gate from `root`, in order, and record how each ended."""
    from unproject.run.manifest import git_commit, git_dirty

    started = datetime.now(UTC)
    results: list[GateResult] = []
    for argv in commands:
        began = time.perf_counter()
        try:
            completed = subprocess.run(
                list(argv), cwd=root, capture_output=True, text=True, check=False
            )
            exit_code = completed.returncode
            output = (completed.stdout or "") + (completed.stderr or "")
        except OSError as error:
            exit_code = 127
            output = f"could not run: {error}"
        results.append(
            GateResult(
                command=" ".join(argv),
                exit_code=exit_code,
                seconds=time.perf_counter() - began,
                output_tail=output[-_TAIL_CHARACTERS:],
            )
        )
    return GatesRecord(
        started_at=started,
        finished_at=datetime.now(UTC),
        git_commit=git_commit(),
        git_dirty=git_dirty(),
        green=all(result.exit_code == 0 for result in results),
        results=results,
    )


def write_gates(record: GatesRecord, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def ci_run_lines(workflow: Path) -> list[str]:
    """Every `run:` command of the CI workflow, in file order.

    A deliberately small reader: the workflow is ours, its steps are one-line
    `run:` entries, and a YAML dependency for this alone would be more code
    than the check it serves.
    """
    lines: list[str] = []
    for raw in workflow.read_text("utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("run:"):
            lines.append(stripped[len("run:") :].strip())
    return lines


__all__ = [
    "GATES",
    "GATES_FILENAME",
    "PYTHON_GATES",
    "VIEWER_GATES",
    "GateResult",
    "GatesRecord",
    "ci_run_lines",
    "run_gates",
    "write_gates",
]
