"""Tests for staged publication and the reuse check.

The stages exercise this through real artifacts; here it is driven directly,
because the interesting cases are the ones where the disk refuses partway
through and are hard to provoke through a whole stage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unproject.run.publish import PublishError, staged_artifact
from unproject.run.reuse import unusable_reason

_DESCRIPTOR = "artifact.json"


def _complete_artifact(directory: Path, *, payload: str = "first") -> None:
    """A directory that reads as finished: a descriptor and a file it points at."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "frame.png").write_text(payload, encoding="utf-8")
    (directory / _DESCRIPTOR).write_text(f'{{"payload": "{payload}"}}', encoding="utf-8")


def test_nothing_appears_at_the_target_until_the_block_ends(tmp_path: Path) -> None:
    target = tmp_path / "run"

    with staged_artifact(target, descriptor=_DESCRIPTOR) as staging:
        (staging / "frame.png").write_text("new", encoding="utf-8")
        assert not target.exists()

    assert (target / "frame.png").read_text("utf-8") == "new"


def test_publication_leaves_no_staging_directory_behind(tmp_path: Path) -> None:
    target = tmp_path / "run"

    with staged_artifact(target, descriptor=_DESCRIPTOR) as staging:
        (staging / _DESCRIPTOR).write_text("{}", encoding="utf-8")

    assert [entry.name for entry in tmp_path.iterdir()] == ["run"]


def test_a_failed_block_publishes_when_there_is_nothing_to_keep(tmp_path: Path) -> None:
    """The manifest of a failed run belongs where the user will look for it."""
    target = tmp_path / "run"

    with (
        pytest.raises(RuntimeError, match="halfway"),
        staged_artifact(target, descriptor=_DESCRIPTOR) as staging,
    ):
        (staging / "manifest.json").write_text('{"status": "failed"}', encoding="utf-8")
        raise RuntimeError("died halfway")

    assert (target / "manifest.json").is_file()
    # No descriptor, so nothing reads it as a finished run.
    assert not (target / _DESCRIPTOR).exists()


def test_a_failed_block_keeps_the_complete_artifact_that_was_there(tmp_path: Path) -> None:
    target = tmp_path / "run"
    _complete_artifact(target)

    with (
        pytest.raises(RuntimeError),
        staged_artifact(target, descriptor=_DESCRIPTOR) as staging,
    ):
        (staging / "frame.png").write_text("second", encoding="utf-8")
        raise RuntimeError("died halfway")

    assert (target / "frame.png").read_text("utf-8") == "first"
    assert (target / _DESCRIPTOR).read_text("utf-8") == '{"payload": "first"}'
    assert [entry.name for entry in tmp_path.iterdir()] == ["run"]


def test_a_successful_block_replaces_a_complete_artifact(tmp_path: Path) -> None:
    target = tmp_path / "run"
    _complete_artifact(target)

    with staged_artifact(target, descriptor=_DESCRIPTOR) as staging:
        _complete_artifact(staging, payload="second")

    assert (target / "frame.png").read_text("utf-8") == "second"
    assert [entry.name for entry in tmp_path.iterdir()] == ["run"]


def test_a_failed_swap_puts_the_previous_artifact_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dangerous window: the old artifact has moved aside, the new one cannot land."""
    target = tmp_path / "run"
    _complete_artifact(target)
    real_replace = Path.replace
    calls: list[Path] = []

    def _second_replace_fails(self: Path, other: str | Path) -> Path:
        calls.append(self)
        # The first call moves the previous artifact aside; the one after it is
        # the new artifact landing, and that is the one that fails here.
        if len(calls) == 2:
            raise OSError("the disk went away")
        return real_replace(self, other)

    monkeypatch.setattr(Path, "replace", _second_replace_fails)

    with (
        pytest.raises(PublishError, match="kept the previous artifact"),
        staged_artifact(target, descriptor=_DESCRIPTOR) as staging,
    ):
        _complete_artifact(staging, payload="second")

    monkeypatch.undo()
    assert (target / "frame.png").read_text("utf-8") == "first"
    assert (target / _DESCRIPTOR).read_text("utf-8") == '{"payload": "first"}'
    assert [entry.name for entry in tmp_path.iterdir()] == ["run"]


def test_a_complete_artifact_has_no_reason_to_be_rejected(tmp_path: Path) -> None:
    _complete_artifact(tmp_path)

    assert unusable_reason(tmp_path, [("frame.png", len("first"))]) is None


def test_a_missing_file_is_named_in_the_reason(tmp_path: Path) -> None:
    reason = unusable_reason(tmp_path, [("frames/000001.png", 10)])

    assert reason is not None
    assert "frames/000001.png" in reason
    assert "not on disk" in reason


def test_a_wrong_size_is_named_with_both_numbers(tmp_path: Path) -> None:
    (tmp_path / "bodies.bin").write_bytes(b"1234")

    reason = unusable_reason(tmp_path, [("bodies.bin", 4096)])

    assert reason is not None
    assert "4 bytes" in reason
    assert "4096" in reason


def test_the_first_problem_is_the_one_reported(tmp_path: Path) -> None:
    """One actionable line beats a list the user has to read through."""
    (tmp_path / "b.png").write_bytes(b"12")

    reason = unusable_reason(tmp_path, [("a.png", 2), ("b.png", 99)])

    assert reason is not None
    assert "a.png" in reason
    assert "b.png" not in reason
