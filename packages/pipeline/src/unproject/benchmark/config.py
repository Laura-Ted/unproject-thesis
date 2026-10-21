"""What a measurement run is, as a versioned file under `experiments/`.

A benchmark names an experiment rather than repeating it. The experiment says
which footage and which estimator settings; this says how to time them.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.config import ConfigError, ExperimentConfig, load_config

BENCHMARK_SCHEMA_VERSION = 1

# What the socket sends, which is the RQ2 output factor. Skeletons are joints
# only; meshes add a rest body per track and a transform per bone per frame.
type OutputMode = Literal["skeleton", "mesh"]


class BenchmarkConfig(BaseModel):
    """One cell of the RQ1/RQ2 matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=BENCHMARK_SCHEMA_VERSION, ge=1)
    # Also the directory raw results land in, hence the character restriction.
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    description: str = ""
    # The experiment being measured, relative to this file.
    experiment: Path
    # Discarded before the steady-state sample starts. The first frames through
    # a freshly built model are not what sustained throughput looks like.
    warmup_frames: int = Field(default=5, ge=0)
    measured_frames: int = Field(default=60, ge=1)
    # ADR-0002 §8.2 asks for at least three where runtime allows.
    repetitions: int = Field(default=3, ge=1)
    output: OutputMode = "skeleton"

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            msg = (
                f"benchmark schema_version {self.schema_version} is not supported "
                f"by this build (expected {BENCHMARK_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Parse and validate a benchmark TOML."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read benchmark config {path}: {error}") from error

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error

    try:
        return BenchmarkConfig.model_validate(data)
    except ValueError as error:
        raise ConfigError(f"{path} failed validation:\n{error}") from error


def load_measured_experiment(
    config: BenchmarkConfig, config_dir: Path
) -> tuple[
    ExperimentConfig,
    Path,
]:
    """The experiment this benchmark measures, and the directory it lives in.

    Returned together because everything downstream resolves the experiment's
    own relative paths against its directory, not the benchmark's.
    """
    path = config.experiment
    resolved = path if path.is_absolute() else (config_dir / path).resolve()
    return load_config(resolved), resolved.parent
