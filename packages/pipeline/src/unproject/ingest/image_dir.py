"""`FrameSource` backed by a directory of already-extracted frames.

Public datasets ship as numbered image folders rather than video, so this is
what lets published scenes be run through the same pipeline as our own captures.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import cv2

from unproject.ingest.hashing import file_sha256
from unproject.ingest.source import FrameSourceError
from unproject.ingest.types import Frame, ImageArray, SourceMetadata

# What OpenCV reads reliably and what the public datasets actually ship.
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

_DIGIT_RUN = re.compile(r"(\d+)")


class ImageDirectoryError(FrameSourceError):
    """Raised when a directory cannot be read as an ordered frame set."""


class ImageDirectorySource:
    """Reads a directory of images as an ordered frame sequence.

    `fps` has to be supplied because images carry no frame rate; it is recorded
    as declared rather than measured. For a genuinely unordered photo collection
    it is a convention and the timestamps mean nothing.
    """

    def __init__(self, directory: Path, *, fps: float, uri: str | None = None) -> None:
        if not directory.is_dir():
            raise ImageDirectoryError(f"no such directory: {directory}")
        if not fps > 0.0:
            raise ImageDirectoryError(f"fps must be positive, got {fps}")

        self._paths = _ordered_images(directory)
        if not self._paths:
            suffixes = ", ".join(sorted(_IMAGE_SUFFIXES))
            raise ImageDirectoryError(f"{directory} holds no image files ({suffixes})")

        self._directory = directory
        self._fps = fps
        self._uri = uri if uri is not None else str(directory)
        self._metadata: SourceMetadata | None = None

    @property
    def directory(self) -> Path:
        return self._directory

    def metadata(self) -> SourceMetadata:
        """Describe the frame set. Cached, since this hashes every file."""
        if self._metadata is None:
            self._metadata = self._probe()
        return self._metadata

    def frames(self) -> Iterator[Frame]:
        """Yield the images in natural filename order."""
        metadata = self.metadata()
        for index, path in enumerate(self._paths):
            image = _read_image(path)
            # Verifying this up front would mean decoding the directory twice,
            # so the check rides along with the read.
            height, width = image.shape[0], image.shape[1]
            if (width, height) != (metadata.width, metadata.height):
                raise ImageDirectoryError(
                    f"{path.name} is {width}x{height} but {self._paths[0].name} is "
                    f"{metadata.width}x{metadata.height}; frames must share a size"
                )
            yield Frame(
                source_index=index,
                timestamp_seconds=index / self._fps,
                image=image,
            )

    def _probe(self) -> SourceMetadata:
        first = _read_image(self._paths[0])
        count = len(self._paths)
        return SourceMetadata(
            kind="image_directory",
            uri=self._uri,
            content_sha256=_frame_set_sha256(self._paths, self._directory),
            width=int(first.shape[1]),
            height=int(first.shape[0]),
            fps=self._fps,
            # Unlike a container's, this count is exact.
            reported_frame_count=count,
            duration_seconds=count / self._fps,
            codec=_image_format(self._paths),
        )


def _ordered_images(directory: Path) -> list[Path]:
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    return sorted(paths, key=_natural_key)


def _natural_key(path: Path) -> tuple[tuple[int, int, str], ...]:
    """Sort digit runs by value, so frame2 comes before frame10.

    Plain lexicographic order silently scrambles any dataset that does not
    zero-pad, and a scrambled frame order is not something later stages notice.
    """
    key: list[tuple[int, int, str]] = []
    for part in _DIGIT_RUN.split(path.name):
        if part.isdigit():
            key.append((0, int(part), ""))
        elif part:
            key.append((1, 0, part))
    return tuple(key)


def _frame_set_sha256(paths: list[Path], directory: Path) -> str:
    """Identity of the whole set: every filename and its contents, in order.

    Renaming a file changes it, which is intended: the order is part of what the
    pipeline consumes.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _image_format(paths: list[Path]) -> str:
    suffixes = {path.suffix.lower().removeprefix(".") for path in paths}
    if len(suffixes) != 1:
        return "mixed"
    suffix = next(iter(suffixes))
    return "jpeg" if suffix == "jpg" else suffix


def _read_image(path: Path) -> ImageArray:
    # IMREAD_COLOR forces 3-channel BGR, matching what the video decoder yields.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageDirectoryError(f"cannot decode {path}")
    return cast(ImageArray, image)
