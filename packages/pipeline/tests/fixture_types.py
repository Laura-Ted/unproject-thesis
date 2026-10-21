"""Call signatures for the shared fixture factories.

Protocols rather than bare `Callable`, so mypy checks the keyword arguments the
tests pass. Otherwise a mistyped keyword only shows up as a failing test.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class MakeVideo(Protocol):
    def __call__(
        self,
        *,
        name: str = ...,
        frame_count: int = ...,
        fps: float = ...,
        width: int = ...,
        height: int = ...,
    ) -> Path: ...


class MakePanningVideo(Protocol):
    def __call__(
        self,
        *,
        name: str = ...,
        frame_count: int = ...,
        step_px: int = ...,
        fps: float = ...,
        width: int = ...,
        height: int = ...,
    ) -> Path: ...


class MakeImageDir(Protocol):
    def __call__(
        self,
        *,
        name: str = ...,
        filenames: Sequence[str] | None = ...,
        frame_count: int = ...,
        width: int = ...,
        height: int = ...,
    ) -> Path: ...


class WriteConfig(Protocol):
    def __call__(self, body: str, *, name: str = ...) -> Path: ...
