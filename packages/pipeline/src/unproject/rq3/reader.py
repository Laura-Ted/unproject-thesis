"""Reading a finished bodies run back out of its artifact.

The viewer does this in TypeScript; analysis needs it in Python. Both slice the
same declared layout rather than assuming one, so a sidecar that does not match
its descriptor is an error here rather than silently reshaped numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from unproject.bodies.artifacts import ARTIFACT_FILENAME, BodiesArtifact

FloatArray = NDArray[np.float32]


class ArtifactReadError(Exception):
    """Raised when a run cannot be read back."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One person in one frame, as the analysis wants them."""

    time_seconds: float
    frame_index: int
    track_id: int
    confidence: float
    # Camera-relative OpenCV metres: +X right, +Y down, +Z forward.
    root_translation: FloatArray
    shape: FloatArray
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Run:
    """A finished bodies run, flattened into what RQ3 measures."""

    artifact: BodiesArtifact
    detections: list[Detection]

    @property
    def track_ids(self) -> list[int]:
        return sorted({detection.track_id for detection in self.detections})

    def track(self, track_id: int) -> list[Detection]:
        """One track's detections, in time order."""
        return [detection for detection in self.detections if detection.track_id == track_id]

    def between(self, start: float, end: float, track_id: int | None) -> list[Detection]:
        """Detections inside a closed time window, optionally for one track.

        Frames where nobody was found are simply absent. Nothing fills them in:
        a missing detection stays missing, per ADR-0002 §8.5.
        """
        return [
            detection
            for detection in self.detections
            if start <= detection.time_seconds <= end
            and (track_id is None or detection.track_id == track_id)
        ]


def read_run(artifact_dir: Path) -> Run:
    """Load a bodies artifact and the sidecar arrays the analysis needs."""
    path = artifact_dir / ARTIFACT_FILENAME
    try:
        artifact = BodiesArtifact.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise ArtifactReadError(f"cannot read {path}: {error}") from error

    translations = read_array(artifact_dir, artifact, "root_translation")
    shapes = read_array(artifact_dir, artifact, "shape")

    detections: list[Detection] = []
    for frame in artifact.frames:
        for person in frame.people:
            index = person.parameter_index
            detections.append(
                Detection(
                    time_seconds=frame.timestamp_seconds,
                    frame_index=frame.frame_index,
                    track_id=person.track_id,
                    confidence=person.confidence,
                    root_translation=translations[index],
                    shape=shapes[index],
                    box_xyxy=person.box_xyxy,
                )
            )
    detections.sort(key=lambda detection: (detection.time_seconds, detection.track_id))
    return Run(artifact=artifact, detections=detections)


def read_array(artifact_dir: Path, artifact: BodiesArtifact, name: str) -> FloatArray:
    """One named array out of the parameter sidecar, in its declared shape."""
    try:
        described = artifact.parameters.named(name)
    except KeyError as error:
        raise ArtifactReadError(str(error)) from error

    path = artifact_dir / artifact.parameters.path
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ArtifactReadError(f"cannot read {path}: {error}") from error

    end = described.byte_offset + described.byte_length
    if len(raw) < end:
        raise ArtifactReadError(
            f"{path} is {len(raw)} bytes; {name} needs {end}. The sidecar is truncated."
        )
    block = np.frombuffer(
        raw, dtype="<f4", count=described.byte_length // 4, offset=described.byte_offset
    )
    return block.reshape(described.shape).astype(np.float32)
