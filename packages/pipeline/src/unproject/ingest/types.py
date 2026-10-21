"""Types shared by every `FrameSource` implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

# BGR, uint8, shape (height, width, 3): OpenCV's native layout, kept as-is.
# Converting to RGB here would cost a full pass over the video, and only the
# consumer knows whether it needs RGB at all.
ImageArray = NDArray[np.uint8]

# Which implementation produced a frame set. Stage [3] cares: video frames are
# temporally ordered and admit sequential matching, a photo collection does not.
SourceKind = Literal["video", "image_directory"]


@dataclass(frozen=True, slots=True)
class Frame:
    """A single decoded frame.

    Not a Pydantic model: it stays in memory, and validating a multi-megabyte
    buffer per frame would cost more than decoding it.
    """

    # Index in the original source, kept even when frames are decimated, so a
    # keyframe can be traced back to the footage.
    source_index: int
    timestamp_seconds: float
    image: ImageArray

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


class SourceMetadata(BaseModel):
    """Identifies the input a run consumed.

    Written to disk, so adding a field is fine but changing what an existing
    one means breaks every reader.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: SourceKind
    # As written in the config, usually relative, so artifacts stay comparable
    # across machines.
    uri: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    # Measured for video, declared by the config for an image directory; `kind`
    # says which, since images carry no frame rate of their own.
    fps: float = Field(gt=0.0)
    # Container metadata, off by a few frames on some VFR footage. Fine for a
    # progress bar, not for a loop bound.
    reported_frame_count: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0.0)
    codec: str
