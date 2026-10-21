"""Stage [1] Ingest: turn a video source into frames plus metadata."""

from unproject.ingest.artifacts import FrameRecord, IngestArtifact, SelectionSummary
from unproject.ingest.image_dir import ImageDirectoryError, ImageDirectorySource
from unproject.ingest.source import FrameSource, FrameSourceError
from unproject.ingest.types import Frame, SourceKind, SourceMetadata
from unproject.ingest.video import VideoDecodeError, VideoFileSource

__all__ = [
    "Frame",
    "FrameRecord",
    "FrameSource",
    "FrameSourceError",
    "ImageDirectoryError",
    "ImageDirectorySource",
    "IngestArtifact",
    "SelectionSummary",
    "SourceKind",
    "SourceMetadata",
    "VideoDecodeError",
    "VideoFileSource",
]
