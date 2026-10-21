"""Tests for `ImageDirectorySource`."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fixture_types import MakeImageDir, MakeVideo
from unproject.ingest import (
    FrameSource,
    ImageDirectoryError,
    ImageDirectorySource,
    VideoFileSource,
)


def test_satisfies_the_frame_source_protocol(make_image_dir: MakeImageDir) -> None:
    source = ImageDirectorySource(make_image_dir(), fps=10.0)

    assert isinstance(source, FrameSource)


def test_metadata_describes_the_frame_set(make_image_dir: MakeImageDir) -> None:
    directory = make_image_dir(frame_count=5, width=64, height=48)

    metadata = ImageDirectorySource(directory, fps=10.0).metadata()

    assert metadata.kind == "image_directory"
    assert (metadata.width, metadata.height) == (64, 48)
    assert metadata.fps == pytest.approx(10.0)
    assert metadata.reported_frame_count == 5
    assert metadata.duration_seconds == pytest.approx(0.5)
    assert metadata.codec == "png"


def test_yields_every_image_with_derived_timestamps(make_image_dir: MakeImageDir) -> None:
    source = ImageDirectorySource(make_image_dir(frame_count=4), fps=8.0)

    frames = list(source.frames())

    assert [frame.source_index for frame in frames] == [0, 1, 2, 3]
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx([0.0, 0.125, 0.25, 0.375])


def _tags(source: ImageDirectorySource) -> list[int]:
    """The per-image marker the fixture writes, read back in yielded order."""
    return [int(frame.image[0, 0, 0]) for frame in source.frames()]


def test_orders_unpadded_filenames_by_number(make_image_dir: MakeImageDir) -> None:
    # The fixture tags each image by its position in this list: 0, 7, 14, 21.
    directory = make_image_dir(filenames=["frame1.png", "frame2.png", "frame10.png", "frame20.png"])

    source = ImageDirectorySource(directory, fps=10.0)

    # Lexicographic order would put frame10 second and scramble the sequence,
    # which nothing downstream would notice.
    assert _tags(source) == [0, 7, 14, 21]


def test_orders_zero_padded_filenames(make_image_dir: MakeImageDir) -> None:
    directory = make_image_dir(filenames=["00002.png", "00010.png", "00001.png"])

    source = ImageDirectorySource(directory, fps=10.0)

    assert _tags(source) == [14, 0, 7]


def test_ignores_non_image_files(make_image_dir: MakeImageDir) -> None:
    directory = make_image_dir(frame_count=3)
    (directory / "poses.txt").write_text("not an image", encoding="utf-8")
    (directory / ".DS_Store").write_bytes(b"\x00")

    source = ImageDirectorySource(directory, fps=10.0)

    assert source.metadata().reported_frame_count == 3


def test_reports_jpeg_as_its_codec(make_image_dir: MakeImageDir) -> None:
    directory = make_image_dir(filenames=["a.jpg", "b.jpg"])

    assert ImageDirectorySource(directory, fps=10.0).metadata().codec == "jpeg"


def test_reports_a_mixed_directory_as_mixed(make_image_dir: MakeImageDir) -> None:
    directory = make_image_dir(filenames=["a.jpg", "b.png"])

    assert ImageDirectorySource(directory, fps=10.0).metadata().codec == "mixed"


def test_content_hash_covers_names_and_contents(make_image_dir: MakeImageDir) -> None:
    directory = make_image_dir(frame_count=3, name="a")
    same = make_image_dir(frame_count=3, name="b")
    renamed = make_image_dir(frame_count=3, name="c")
    (renamed / "00002.png").rename(renamed / "00009.png")

    digest = ImageDirectorySource(directory, fps=10.0).metadata().content_sha256

    assert ImageDirectorySource(same, fps=10.0).metadata().content_sha256 == digest
    # The order is part of what the pipeline consumes, so a rename is a change.
    assert ImageDirectorySource(renamed, fps=10.0).metadata().content_sha256 != digest


def test_is_re_iterable(make_image_dir: MakeImageDir) -> None:
    source = ImageDirectorySource(make_image_dir(frame_count=3), fps=10.0)

    assert len(list(source.frames())) == len(list(source.frames())) == 3


def test_frames_carry_a_bgr_uint8_image(make_image_dir: MakeImageDir) -> None:
    source = ImageDirectorySource(make_image_dir(width=64, height=48), fps=10.0)

    frame = next(iter(source.frames()))

    assert frame.image.shape == (48, 64, 3)
    assert frame.image.dtype.name == "uint8"


def test_uri_is_recorded_as_declared(make_image_dir: MakeImageDir) -> None:
    directory = make_image_dir()

    metadata = ImageDirectorySource(directory, fps=10.0, uri="datasets/garden").metadata()

    assert metadata.uri == "datasets/garden"


def test_missing_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ImageDirectoryError, match="no such directory"):
        ImageDirectorySource(tmp_path / "absent", fps=10.0)


def test_empty_directory_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(ImageDirectoryError, match="no image files"):
        ImageDirectorySource(tmp_path / "empty", fps=10.0)


def test_non_positive_fps_is_rejected(make_image_dir: MakeImageDir) -> None:
    with pytest.raises(ImageDirectoryError, match="fps must be positive"):
        ImageDirectorySource(make_image_dir(), fps=0.0)


def test_mismatched_frame_sizes_are_rejected(make_image_dir: MakeImageDir) -> None:
    directory = make_image_dir(frame_count=2, width=64, height=48)
    odd = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(directory / "00002.png"), odd)

    source = ImageDirectorySource(directory, fps=10.0)

    with pytest.raises(ImageDirectoryError, match="frames must share a size"):
        list(source.frames())


def test_undecodable_image_is_rejected(make_image_dir: MakeImageDir) -> None:
    directory = make_image_dir(frame_count=1)
    (directory / "00001.png").write_bytes(b"not a png")

    source = ImageDirectorySource(directory, fps=10.0)

    with pytest.raises(ImageDirectoryError, match="cannot decode"):
        list(source.frames())


def test_both_sources_fill_the_same_metadata_contract(
    make_image_dir: MakeImageDir, make_video: MakeVideo
) -> None:
    """The point of the protocol: one consumer, two shapes of input."""
    sources: list[FrameSource] = [
        ImageDirectorySource(make_image_dir(frame_count=4, width=64, height=48), fps=10.0),
        VideoFileSource(make_video(frame_count=4, fps=10.0, width=64, height=48)),
    ]

    described = [source.metadata() for source in sources]

    assert [metadata.kind for metadata in described] == ["image_directory", "video"]
    for metadata in described:
        assert (metadata.width, metadata.height) == (64, 48)
        assert metadata.fps == pytest.approx(10.0)
        assert metadata.reported_frame_count == 4
        assert len(metadata.content_sha256) == 64
    assert [len(list(source.frames())) for source in sources] == [4, 4]
