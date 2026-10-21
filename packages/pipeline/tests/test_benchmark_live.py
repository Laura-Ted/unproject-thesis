"""Tests for the live report: what a row means, and what the summary makes of it.

ADR-0005 WP3. The rule under test is that every interval is measured on one
clock - a client's from send to receive and to present, the server's for its
own phases - and that a stored summary cannot disagree with its own rows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from unproject.benchmark.live import (
    LiveConfig,
    LiveEnvironment,
    LiveReport,
    LiveSample,
    ServerTiming,
    read_reports,
    summarise_live,
    write_report,
)

_STARTED = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def _sample(
    frame_id: int,
    sent: float,
    received: float | None = None,
    presented: float | None = None,
    *,
    dropped: int | None = 0,
    inference: float = 300.0,
) -> LiveSample:
    return LiveSample(
        frame_id=frame_id,
        sent_at_ms=sent,
        sent_bytes=1000,
        received_at_ms=received,
        presented_at_ms=presented,
        received_bytes=None if received is None else 500,
        people=None if received is None else 3,
        dropped_before=None if received is None else dropped,
        server=(
            None
            if received is None
            else ServerTiming(decode_ms=2.0, queue_wait_ms=10.0, inference_ms=inference)
        ),
    )


def _report(samples: list[LiveSample], **overrides: object) -> LiveReport:
    defaults: dict[str, object] = {
        "kind": "transport",
        "started_at": _STARTED,
        "source": "datasets/pedestrians/clip.mp4",
        "source_sha256": "a" * 64,
        "mode": "skeleton",
        "frame_width": 1280,
        "frame_height": 720,
        "environment": LiveEnvironment(client_platform="test"),
        "config": LiveConfig(send_fps=2.0, max_width=1280, jpeg_quality=0.75, bodies=False),
        "samples": samples,
        "summary": summarise_live(samples),
    }
    return LiveReport.model_validate(defaults | overrides)


class TestASample:
    def test_an_answer_cannot_precede_its_question(self) -> None:
        with pytest.raises(ValueError, match="received before it was sent"):
            _sample(0, sent=100.0, received=50.0)

    def test_presentation_needs_a_reception_and_follows_it(self) -> None:
        with pytest.raises(ValueError, match="without being received"):
            LiveSample(frame_id=0, sent_at_ms=0.0, sent_bytes=1, presented_at_ms=5.0)
        with pytest.raises(ValueError, match="presented before it was received"):
            _sample(0, sent=0.0, received=100.0, presented=90.0)

    def test_an_unanswered_frame_is_a_row_not_a_gap(self) -> None:
        sample = _sample(3, sent=1500.0)

        assert sample.received_at_ms is None
        assert sample.server is None


class TestTheSummary:
    def test_round_trip_and_present_are_on_the_client_clock(self) -> None:
        summary = summarise_live(
            [
                _sample(0, sent=0.0, received=400.0, presented=416.0),
                _sample(1, sent=500.0, received=850.0, presented=870.0),
                _sample(2, sent=1000.0, received=1500.0, presented=1510.0),
            ]
        )

        assert summary.round_trip_ms.median == 400.0
        assert summary.round_trip_ms.max == 500.0
        assert summary.capture_to_present_ms.median == 416.0
        assert summary.receive_to_present_ms.median == 16.0
        # Server phases are the server's numbers, never a difference of clocks.
        assert summary.inference_ms.median == 300.0
        assert summary.queue_wait_ms.median == 10.0

    def test_drops_are_the_servers_count_and_the_rest_is_unaccounted(self) -> None:
        """Nine sent at 8 fps into a 2 fps estimator: the server answers a
        few, reports what it replaced, and whatever was in flight at the end
        is neither answered nor reported."""
        summary = summarise_live(
            [
                _sample(0, sent=0.0, received=450.0, dropped=0),
                _sample(1, sent=125.0),
                _sample(2, sent=250.0),
                _sample(3, sent=375.0),
                _sample(4, sent=500.0, received=960.0, dropped=3),
                _sample(5, sent=625.0),
                _sample(6, sent=750.0),
                _sample(7, sent=875.0, received=1400.0, dropped=2),
                _sample(8, sent=1000.0),
            ]
        )

        assert summary.frames_sent == 9
        assert summary.frames_answered == 3
        assert summary.frames_dropped_by_server == 5
        assert summary.frames_unaccounted == 1
        assert summary.duration_ms == 1400.0
        assert summary.answered_fps == pytest.approx(3 / 1.4, rel=1e-3)

    def test_bytes_each_way_are_summed(self) -> None:
        summary = summarise_live([_sample(0, 0.0, 100.0), _sample(1, 50.0)])

        assert summary.bytes_sent == 2000
        assert summary.bytes_received == 500

    def test_nothing_answered_leaves_the_distributions_empty(self) -> None:
        summary = summarise_live([_sample(0, 0.0), _sample(1, 100.0)])

        assert summary.frames_answered == 0
        assert summary.round_trip_ms.samples == 0
        assert summary.round_trip_ms.median is None
        # Frames were sent over 100 ms and none came back: a measured zero,
        # not a missing number.
        assert summary.answered_fps == 0.0

    def test_the_same_rows_give_the_same_summary(self) -> None:
        rows = [_sample(0, 0.0, 300.0), _sample(1, 500.0, 900.0)]

        assert summarise_live(rows) == summarise_live(list(reversed(rows)))


class TestTheReport:
    def test_a_stored_summary_must_match_its_rows(self) -> None:
        rows = [_sample(0, 0.0, 300.0)]
        stale = summarise_live([_sample(0, 0.0, 900.0)])

        with pytest.raises(ValueError, match="does not match its own samples"):
            _report(rows, summary=stale)

    def test_a_frame_id_cannot_appear_twice(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            _report([_sample(0, 0.0, 300.0), _sample(0, 500.0, 800.0)])

    def test_an_unreadable_schema_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            _report([_sample(0, 0.0, 300.0)], schema_version=99)

    def test_it_is_written_once_and_read_back(self, tmp_path: Path) -> None:
        report = _report([_sample(0, 0.0, 300.0), _sample(1, 500.0)])

        path = write_report(report, tmp_path)
        reports, unreadable = read_reports(tmp_path / "live")

        assert path.name.endswith("-transport-skeleton-2fps.json")
        assert unreadable == []
        assert reports == [report]

    def test_a_corrupt_report_is_named_not_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "live").mkdir()
        (tmp_path / "live" / "broken.json").write_text("{ no", "utf-8")

        reports, unreadable = read_reports(tmp_path / "live")

        assert reports == []
        assert unreadable and "broken.json" in unreadable[0]
