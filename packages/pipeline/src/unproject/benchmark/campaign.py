"""A campaign: the same suites, under the same protocol, on declared machines.

ADR-0006. Protocol v2 measures one machine at a time and says so in every
result; a campaign is the set of such runs the thesis compares. Two things are
fixed here so the comparison is one:

- **the machines are declared before they measure**, in a registry
  (`benchmarks/machines.toml`): an id, what it is, which accelerator it
  offers. A job records the id it ran under, and the summary marks a cell
  whose jobs name two ids as not meeting the protocol;
- **each machine writes under its own root**, `<campaign>/<machine id>/`, so
  results from two machines cannot land in one directory and be averaged by
  a summary that does not know better.

This module reads the registry and folds the per-machine summaries into one
table. Like the suite summary it runs no inference and reads only what the
runs wrote; the same inputs produce the same bytes.
"""

from __future__ import annotations

import csv
import io
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.benchmark.suite import STAGE_NAME
from unproject.benchmark.summary import SUMMARY_FILENAME, CellSummary, PairSummary, SuiteSummary
from unproject.bodies.device import DeviceKind
from unproject.config import ConfigError

CAMPAIGN_SCHEMA_VERSION = 1
CAMPAIGN_FILENAME = "campaign.json"
CAMPAIGN_CSV_FILENAME = "campaign.csv"
CAMPAIGN_PAIRS_CSV_FILENAME = "campaign-pairs.csv"

# Where the registry lives, relative to the repository root.
DEFAULT_REGISTRY = Path("benchmarks/machines.toml")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Machine(_Frozen):
    """One declared machine of the campaign."""

    # Also a directory name under the campaign root, hence the restriction.
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    # Prose: the model, the chip, the memory, whose it is.
    label: str = Field(min_length=1)
    # The accelerator the machine offers. A CPU-only suite may run on any
    # machine; an `auto` suite on this one is expected to find this kind.
    device: DeviceKind
    # Which class of hardware the thesis counts it as. Consumer is what the
    # research question is about; a rented host is consumer silicon on a
    # server, and says so.
    hardware_class: Literal["consumer", "rented-consumer-gpu", "server", "other"] = "consumer"
    notes: str = ""


class MachineRegistry(_Frozen):
    schema_version: int = Field(default=CAMPAIGN_SCHEMA_VERSION, ge=1)
    machines: list[Machine] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        ids = [machine.id for machine in self.machines]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            msg = f"machines declared more than once: {duplicates}"
            raise ValueError(msg)
        return self

    def find(self, machine_id: str) -> Machine | None:
        for machine in self.machines:
            if machine.id == machine_id:
                return machine
        return None


def load_registry(path: Path) -> MachineRegistry:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read machine registry {path}: {error}") from error
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error
    try:
        return MachineRegistry.model_validate(data)
    except ValueError as error:
        raise ConfigError(f"{path} failed validation:\n{error}") from error


def machine_root(campaign_root: Path, machine_id: str) -> Path:
    """The artifacts root one machine's runs of a campaign write under."""
    return campaign_root / machine_id


class CampaignCell(_Frozen):
    """One cell of one suite on one machine: the suite summary's row, plus
    where it came from."""

    machine_id: str
    suite: str
    cell: CellSummary
    # The machine the results themselves recorded. Equal to `machine_id`
    # unless something was moved; then both are shown and `consistent` is
    # false.
    recorded_machine_ids: list[str]
    consistent: bool


class CampaignPair(_Frozen):
    machine_id: str
    suite: str
    pair: PairSummary


class CampaignSummary(_Frozen):
    schema_version: int = Field(default=CAMPAIGN_SCHEMA_VERSION, ge=1)
    root: str
    machines: list[str]
    suites: list[str]
    cells: list[CampaignCell]
    pairs: list[CampaignPair]
    # Directories under the root that hold no readable summary, or whose
    # machine is not in the registry, named rather than skipped.
    missing: list[str] = Field(default_factory=list)


def read_campaign(root: Path, *, registry: MachineRegistry | None = None) -> CampaignSummary:
    """Every `<machine>/benchmark-v2/<suite>/summary.json` under the root.

    Sorted by machine then suite then cell so the output does not depend on
    directory order. A machine directory that is not in the registry is
    still read - the registry is checked, not enforced, here - and named in
    `missing`, as is anything that holds no readable summary.
    """
    if not root.is_dir():
        raise ConfigError(f"no campaign at {root}")
    cells: list[CampaignCell] = []
    pairs: list[CampaignPair] = []
    missing: list[str] = []
    machines: list[str] = []
    suites: set[str] = set()
    for machine_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        machine_id = machine_dir.name
        if registry is not None and registry.find(machine_id) is None:
            missing.append(f"{machine_id}: not in the registry")
        stage_dir = machine_dir / STAGE_NAME
        suite_dirs = (
            sorted(p for p in stage_dir.iterdir() if p.is_dir()) if stage_dir.is_dir() else []
        )
        found = False
        for suite_dir in suite_dirs:
            summary_path = suite_dir / SUMMARY_FILENAME
            if not summary_path.is_file():
                missing.append(f"{machine_id}/{suite_dir.name}: no {SUMMARY_FILENAME}")
                continue
            try:
                summary = SuiteSummary.model_validate_json(summary_path.read_text("utf-8"))
            except (OSError, ValueError) as error:
                missing.append(f"{machine_id}/{suite_dir.name}: unreadable summary: {error}")
                continue
            found = True
            suites.add(summary.suite)
            for cell in summary.cells:
                recorded = list(cell.machine_ids)
                cells.append(
                    CampaignCell(
                        machine_id=machine_id,
                        suite=summary.suite,
                        cell=cell,
                        recorded_machine_ids=recorded,
                        consistent=recorded == [machine_id] and cell.one_machine,
                    )
                )
            pairs.extend(
                CampaignPair(machine_id=machine_id, suite=summary.suite, pair=pair)
                for pair in summary.pairs
            )
        if found:
            machines.append(machine_id)
        elif not suite_dirs:
            missing.append(f"{machine_id}: no results")
    return CampaignSummary(
        root=str(root),
        machines=machines,
        suites=sorted(suites),
        cells=cells,
        pairs=pairs,
        missing=missing,
    )


def to_csv(summary: CampaignSummary) -> str:
    """One row per machine, suite and cell: the columns a thesis table needs."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "machine_id",
            "suite",
            "cell",
            "output",
            "device",
            "device_name",
            "machine",
            "consistent",
            "meets_protocol",
            "jobs",
            "samples",
            "inference_median_s",
            "inference_p25_s",
            "inference_p75_s",
            "inference_p95_s",
            "inference_median_per_job_s",
            "total_median_s",
            "throughput_fps_median",
            "model_load_s",
            "people_median",
            "peak_rss_bytes",
            "model_footprint_bytes",
            "metal_driver_peak_bytes",
            "cuda_reserved_peak_bytes",
            "source_size",
            "model_input_size",
            "load_average_max",
            "git_commit",
            "git_dirty",
        ]
    )
    for entry in summary.cells:
        cell = entry.cell
        writer.writerow(
            [
                entry.machine_id,
                entry.suite,
                cell.name,
                cell.output,
                ";".join(cell.devices),
                ";".join(cell.device_names),
                ";".join(cell.machines),
                str(entry.consistent).lower(),
                str(cell.meets_protocol).lower(),
                cell.jobs,
                cell.inference.samples,
                _cell(cell.inference.median),
                _cell(cell.inference.p25),
                _cell(cell.inference.p75),
                _cell(cell.inference.p95),
                ";".join(f"{value:.6f}" for value in cell.inference_median_per_job),
                _cell(cell.total.median),
                _cell(cell.throughput_fps_median),
                ";".join(f"{value:.6f}" for value in cell.model_load_seconds),
                _cell(cell.people.median),
                _cell(cell.peak_rss_bytes),
                _cell(cell.model_footprint_bytes),
                _cell(cell.metal_driver_peak_bytes),
                _cell(cell.cuda_reserved_peak_bytes),
                cell.source_size,
                cell.model_input_size,
                _cell(cell.load_average_max),
                ";".join(commit[:12] for commit in cell.git_commits),
                str(cell.git_dirty).lower(),
            ]
        )
    return buffer.getvalue()


def pairs_to_csv(summary: CampaignSummary) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "machine_id",
            "suite",
            "a",
            "b",
            "question",
            "pairs",
            "same_frames",
            "difference_median_s",
            "fraction_b_slower",
            "median_ratio",
        ]
    )
    for entry in summary.pairs:
        pair = entry.pair
        writer.writerow(
            [
                entry.machine_id,
                entry.suite,
                pair.a,
                pair.b,
                pair.question,
                pair.pairs,
                str(pair.same_frames).lower(),
                _cell(pair.difference.median),
                _cell(pair.fraction_b_slower),
                _cell(pair.median_ratio),
            ]
        )
    return buffer.getvalue()


def _cell(value: float | int | None) -> str:
    return "" if value is None else str(value)


def write_campaign(summary: CampaignSummary, root: Path) -> tuple[Path, Path, Path]:
    as_json = root / CAMPAIGN_FILENAME
    as_csv = root / CAMPAIGN_CSV_FILENAME
    as_pairs = root / CAMPAIGN_PAIRS_CSV_FILENAME
    as_json.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    as_csv.write_text(to_csv(summary), encoding="utf-8")
    as_pairs.write_text(pairs_to_csv(summary), encoding="utf-8")
    return as_json, as_csv, as_pairs


__all__ = [
    "CAMPAIGN_CSV_FILENAME",
    "CAMPAIGN_FILENAME",
    "CAMPAIGN_PAIRS_CSV_FILENAME",
    "DEFAULT_REGISTRY",
    "CampaignCell",
    "CampaignPair",
    "CampaignSummary",
    "Machine",
    "MachineRegistry",
    "load_registry",
    "machine_root",
    "pairs_to_csv",
    "read_campaign",
    "to_csv",
    "write_campaign",
]
