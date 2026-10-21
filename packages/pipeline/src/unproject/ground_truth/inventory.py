"""What is actually on disk, before anything tries to read it.

Two jobs. It fails fast when a release is absent or shaped unexpectedly, saying
what was expected and what to do — a licensed dataset is the author's to fetch,
so the error has to be actionable by a person rather than by a retry.

And it computes the content identity that goes into the evaluation descriptor.
Over the annotation files and the subset manifest, never over the images: the
images are large (122 GB for Anny-One) and the annotations are what a metric
reads.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from unproject.ground_truth.config import DatasetKind, GroundTruthConfig
from unproject.ground_truth.dataset import GroundTruthError

# Read in chunks so hashing does not load a file into memory.
_CHUNK_BYTES = 1024 * 1024

# The annotation formats each release is expected to carry. Images are excluded
# deliberately; see the module docstring.
_ANNOTATION_SUFFIXES = frozenset({".mat", ".json", ".npz", ".csv", ".txt", ".toml", ".yaml"})

# Where to get a dataset that is missing. ADR-0004 §4.1 leaves acquisition to
# the author, so the message names the official page.
_WHERE: dict[DatasetKind, str] = {
    "mupots-3d": "https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/",
    "anny-one": (
        "https://europe.naverlabs.com/research/"
        "vision-foundation-models-for-human-understanding/anny-one/"
    ),
}


@dataclass(frozen=True, slots=True)
class Inventory:
    """The integrity record ADR-0004 §4.3 asks for."""

    dataset: DatasetKind
    root: Path
    files: tuple[Path, ...]
    file_count: int
    byte_count: int
    content_sha256: str


def inventory(config: GroundTruthConfig, *, dataset_root: Path) -> Inventory:
    """Walk the release and identify it by content.

    Raises rather than returning an empty inventory: an evaluation over nothing
    would otherwise report perfect coverage of zero people.
    """
    if not dataset_root.exists():
        raise GroundTruthError(
            f"{config.dataset} is not at {dataset_root}. Obtain it from "
            f"{_WHERE[config.dataset]}, accept its licence, and place it there. "
            "Nothing in this repository will fetch it for you: the licence is "
            "yours to accept."
        )
    if not dataset_root.is_dir():
        raise GroundTruthError(f"{dataset_root} is a file; the dataset root must be a directory")

    files = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _ANNOTATION_SUFFIXES
    )
    if not files:
        raise GroundTruthError(
            f"{dataset_root} holds no annotation files. Looked for "
            f"{sorted(_ANNOTATION_SUFFIXES)} anywhere beneath it and found only "
            f"{_describe_contents(dataset_root)}. Check the archive was extracted "
            "and that this is the directory it was extracted into."
        )

    digest = hashlib.sha256()
    byte_count = 0
    for path in files:
        # The relative path goes into the digest too, so moving a file within
        # the release changes the identity even when its bytes do not.
        digest.update(path.relative_to(dataset_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(path).encode("ascii"))
        digest.update(b"\0")
        byte_count += path.stat().st_size

    return Inventory(
        dataset=config.dataset,
        root=dataset_root,
        files=tuple(files),
        file_count=len(files),
        byte_count=byte_count,
        content_sha256=digest.hexdigest(),
    )


def read_subset_manifest(path: Path) -> tuple[tuple[str, ...], str]:
    """The frozen list of sample identifiers, and the digest of the file.

    Committed before inference and never regenerated afterwards: a subset chosen
    once predictions are visible is a subset chosen for its answer.
    """
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise GroundTruthError(
            f"cannot read subset manifest {path}: {error}. Anny-One requires one; "
            "generate it deterministically and commit it before running inference."
        ) from error

    lines = raw.decode("utf-8").splitlines()
    identifiers = tuple(
        stripped for line in lines if (stripped := line.strip()) and not stripped.startswith("#")
    )
    if not identifiers:
        raise GroundTruthError(f"{path} lists no sample identifiers")

    duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
    if duplicates:
        raise GroundTruthError(
            f"{path} lists {duplicates} more than once, which would weight those "
            "samples twice in every average"
        )
    if list(identifiers) != sorted(identifiers):
        raise GroundTruthError(
            f"{path} is not sorted. A stable order is what makes the manifest "
            "reproducible; sort it before committing."
        )
    return identifiers, hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_contents(root: Path) -> str:
    """A short summary of what is there, for an error worth acting on."""
    suffixes = sorted({path.suffix.lower() for path in root.rglob("*") if path.is_file()})
    if not suffixes:
        return "an empty directory"
    return f"file types {suffixes}"
