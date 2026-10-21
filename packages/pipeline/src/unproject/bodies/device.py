"""Which device the model runs on, chosen once and written down.

The order of preference is CUDA, then Metal, then the CPU, unless the
environment variable `UNPROJECT_DEVICE` names one, in which case it is that one
or an error. Never a fallback: a benchmark that asked for the GPU and quietly
got the CPU would be a number about the wrong thing, and the same holds for a
Linux box whose driver is too old for the installed PyTorch.

Everything torch touches is imported inside the function that needs it: this
module has to stay importable without torch, because the CLI reaches pycolmap
too and the two cannot share a process (docs/reproduce.md §6).
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

type DeviceKind = Literal["cuda", "mps", "cpu"]
# Also the order of preference.
DEVICE_KINDS: tuple[DeviceKind, ...] = ("cuda", "mps", "cpu")

# Set it to one of `DEVICE_KINDS` to insist; unset or "auto" to take the best.
ENV_DEVICE = "UNPROJECT_DEVICE"
AUTO: Literal["auto"] = "auto"

type DeviceRequest = DeviceKind | Literal["auto"]


class DeviceError(Exception):
    """Raised when the device that was asked for cannot be used."""


class DeviceInfo(BaseModel):
    """The device as PyTorch and the driver name it. Recorded with every result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: DeviceKind
    # The accelerator's own name ("NVIDIA GeForce RTX 4070"); on Metal and on
    # the CPU there is no separate part to name, so this repeats the chip.
    name: str
    torch_version: str
    # CUDA only. The toolkit the wheel was built against, the cuDNN it loaded,
    # and the driver on the host, which is the one that can be too old.
    cuda_version: str | None = None
    cudnn_version: str | None = None
    driver_version: str | None = None
    # Intra-op threads torch will use. Decides a CPU number more than anything.
    threads: int = Field(ge=1)


def requested_kind(env: Mapping[str, str] | None = None) -> DeviceRequest:
    """What the environment asks for: a kind, or `auto`."""
    value = (os.environ if env is None else env).get(ENV_DEVICE, AUTO).strip().lower()
    if value in {"", AUTO}:
        return AUTO
    if value not in DEVICE_KINDS:
        raise DeviceError(
            f"{ENV_DEVICE}={value!r} is not a device; use one of {', '.join(DEVICE_KINDS)} or auto"
        )
    # `value` is one of the literals; the check above is the narrowing.
    kind: DeviceKind = "cuda" if value == "cuda" else "mps" if value == "mps" else "cpu"
    return kind


def available_kinds() -> list[DeviceKind]:
    """Every kind this PyTorch can run on, best first. The CPU is always one."""
    import torch

    kinds: list[DeviceKind] = []
    if torch.cuda.is_available():
        kinds.append("cuda")
    if torch.backends.mps.is_available():
        kinds.append("mps")
    kinds.append("cpu")
    return kinds


def resolve_kind(requested: DeviceRequest | None = None) -> DeviceKind:
    """The kind that will run, given what was asked and what exists.

    `auto` takes the first available. A named kind that is not available is an
    error rather than a fallback, so a result never records a device other than
    the one that was asked for.
    """
    wanted = requested_kind() if requested is None else requested
    available = available_kinds()
    if wanted == AUTO:
        chosen = available[0]
        if chosen == "cpu":
            logger.warning("no CUDA or Metal backend available; the model will run on the CPU")
        return chosen
    if wanted not in available:
        raise DeviceError(
            f"{ENV_DEVICE}={wanted} was asked for, but this PyTorch only has "
            f"{', '.join(available)}. On Linux with an NVIDIA card, check `nvidia-smi` and "
            "that the driver is new enough for the CUDA the installed torch was built with."
        )
    return wanted


def select_device(requested: DeviceRequest | None = None) -> torch.device:
    """`resolve_kind` as a `torch.device`."""
    import torch

    return torch.device(resolve_kind(requested))


def describe_device(device: torch.device) -> DeviceInfo:
    """Name the device and the software under it, for the record."""
    import torch

    kind = _kind_of(device)
    name = chip_name()
    cuda_version: str | None = None
    cudnn_version: str | None = None
    driver_version: str | None = None
    if kind == "cuda":
        name = str(torch.cuda.get_device_name(device))
        cuda_version = torch.version.cuda
        cudnn_version = _cudnn_version()
        driver_version = nvidia_driver_version()
    return DeviceInfo(
        kind=kind,
        name=name,
        torch_version=str(torch.__version__),
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        driver_version=driver_version,
        threads=max(1, int(torch.get_num_threads())),
    )


def synchronize(device: torch.device) -> None:
    """Wait for the device's queue to drain, so a clock stopped after this
    has seen the whole frame and not just its submission."""
    import torch

    kind = _kind_of(device)
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()


def accelerator_memory(device: torch.device) -> tuple[int | None, int | None]:
    """What the accelerator holds now: allocated by tensors, and held from the
    system by the allocator (Metal: the driver figure; CUDA: reserved).

    Absent, not zero, on the CPU and where the installed PyTorch lacks the
    counter.
    """
    import torch

    kind = _kind_of(device)
    if kind == "cuda":
        return int(torch.cuda.memory_allocated(device)), int(torch.cuda.memory_reserved(device))
    if kind == "mps":
        allocated = getattr(torch.mps, "current_allocated_memory", None)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        return (
            int(allocated()) if callable(allocated) else None,
            int(driver()) if callable(driver) else None,
        )
    return None, None


def accelerator_peak_memory(device: torch.device) -> tuple[int | None, int | None]:
    """The process high-water marks CUDA keeps itself; Metal keeps none, so a
    caller sampling `accelerator_memory` after every frame is the peak there."""
    import torch

    if _kind_of(device) != "cuda":
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _cudnn_version() -> str | None:
    """cuDNN as loaded, or None. Untyped upstream, so the read is fenced here."""
    import torch

    version = cast("Callable[[], int | None]", torch.backends.cudnn.version)()
    return None if version is None else str(version)


def nvidia_driver_version() -> str | None:
    """The host driver, from `nvidia-smi`; None where there is none to ask."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    first = completed.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _kind_of(device: torch.device) -> DeviceKind:
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    return "cpu"


def chip_name() -> str:
    """The CPU, as the OS names it: `sysctl` on macOS, `/proc/cpuinfo` on Linux."""
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


__all__ = [
    "AUTO",
    "DEVICE_KINDS",
    "ENV_DEVICE",
    "DeviceError",
    "DeviceInfo",
    "DeviceKind",
    "DeviceRequest",
    "accelerator_memory",
    "accelerator_peak_memory",
    "available_kinds",
    "chip_name",
    "describe_device",
    "nvidia_driver_version",
    "requested_kind",
    "resolve_kind",
    "select_device",
    "synchronize",
]
