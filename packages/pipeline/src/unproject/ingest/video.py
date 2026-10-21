"""`FrameSource` backed by a local video file."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import cv2

from unproject.ingest.hashing import file_sha256
from unproject.ingest.source import FrameSourceError
from unproject.ingest.types import Frame, ImageArray, SourceMetadata

logger = logging.getLogger(__name__)


class VideoDecodeError(FrameSourceError):
    """Raised when a video cannot be opened or its metadata is unusable."""


class VideoFileSource:
    """Lazily decodes a video file frame by frame.

    `frames()` opens and releases its own capture, so the source can be iterated
    more than once and an abandoned iterator does not leak a file handle.
    """

    def __init__(self, path: Path, *, uri: str | None = None) -> None:
        if not path.is_file():
            raise VideoDecodeError(f"no such video file: {path}")
        self._path = path
        # Decoding needs the resolved path, but artifacts should record the one
        # the config declared.
        self._uri = uri if uri is not None else str(path)
        self._metadata: SourceMetadata | None = None

    @property
    def path(self) -> Path:
        return self._path

    def metadata(self) -> SourceMetadata:
        """Probe the container. Cached, since this hashes the whole file."""
        if self._metadata is None:
            self._metadata = self._probe()
        return self._metadata

    def frames(self) -> Iterator[Frame]:
        """Yield every frame in decode order, oldest first.

        Timestamps come from the container where usable, otherwise from the
        nominal frame rate. The container is preferred because handheld capture
        is often variable-rate, where `index / fps` drifts.
        """
        fps = self.metadata().fps
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise VideoDecodeError(f"OpenCV cannot open {self._path}")

        try:
            index = 0
            last_timestamp = -1.0
            warned = False
            while True:
                ok, image = capture.read()
                if not ok:
                    break

                # Read the position after decoding, where it refers to the frame
                # just returned. Reading it before is off by one and shifts the
                # whole timeline without failing anywhere.
                container_seconds = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if container_seconds > last_timestamp:
                    timestamp = container_seconds
                else:
                    # Some containers report 0 ms for every frame. The fallback is
                    # only correct at constant rate, so warn rather than pretend.
                    timestamp = index / fps
                    if not warned:
                        logger.warning(
                            "%s reports unusable frame timestamps; "
                            "falling back to the nominal %.3f fps",
                            self._path,
                            fps,
                        )
                        warned = True

                yield Frame(
                    source_index=index,
                    timestamp_seconds=timestamp,
                    # read() always returns uint8 BGR; the stubs say ndarray.
                    image=cast(ImageArray, image),
                )
                last_timestamp = timestamp
                index += 1
        finally:
            capture.release()

    def _probe(self) -> SourceMetadata:
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise VideoDecodeError(f"OpenCV cannot open {self._path}")

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        finally:
            capture.release()

        if width <= 0 or height <= 0:
            raise VideoDecodeError(f"{self._path} reports a {width}x{height} frame size")
        # A zero or NaN fps makes every downstream timestamp meaningless, and
        # pose estimation would degrade instead of failing. Refuse it here.
        if not fps > 0.0 or fps != fps:
            raise VideoDecodeError(f"{self._path} reports an unusable frame rate: {fps}")

        duration = reported_count / fps if reported_count > 0 else None

        return SourceMetadata(
            kind="video",
            uri=self._uri,
            content_sha256=file_sha256(self._path),
            width=width,
            height=height,
            fps=fps,
            reported_frame_count=reported_count if reported_count > 0 else None,
            duration_seconds=duration,
            codec=_fourcc_to_str(fourcc),
        )


def _fourcc_to_str(fourcc: int) -> str:
    """Unpack OpenCV's four-character codec code."""
    if fourcc <= 0:
        return "unknown"
    code = "".join(chr((fourcc >> (8 * shift)) & 0xFF) for shift in range(4))
    return code.strip() or "unknown"
