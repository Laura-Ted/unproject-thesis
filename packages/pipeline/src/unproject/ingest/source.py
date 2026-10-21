"""The interface every input goes through."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from unproject.ingest.types import Frame, SourceMetadata


class FrameSourceError(Exception):
    """Raised when a source cannot be opened or read.

    Shared base so callers can handle any input failure in one place.
    """


@runtime_checkable
class FrameSource(Protocol):
    """Yields frames with timestamps from any source.

    Implementations should be lazy: callers may stop early, and the whole video
    is not expected to fit in memory.

    Implemented: `VideoFileSource`, `ImageDirectorySource`. An RTSP camera
    source is deferred until a declared experiment needs one (ADR-0002 §6.3).
    """

    def frames(self) -> Iterator[Frame]: ...

    def metadata(self) -> SourceMetadata: ...
