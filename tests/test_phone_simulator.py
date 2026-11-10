"""Tests for basir/api/phone_simulator.py: record the toy dataset, stream it as two phones to a live server."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from basir.api.phone_simulator import PhoneStats, record, simulate
from basir.api.protocol import decode_frame, encode_frame
from basir.ingest.datasets.toy import create_toy_dataset
from basir.ingest.sources.tum import TUMDatasetSource


@pytest.fixture
def toy_frames(tmp_path: Path) -> list:
    return record(TUMDatasetSource(create_toy_dataset(tmp_path / "toy", num_frames=4)))


def test_record_encodes_jpeg_pose_and_intrinsics(toy_frames: list) -> None:
    assert len(toy_frames) == 4
    first = toy_frames[0]
    header, jpeg = decode_frame(encode_frame(first.header, first.jpeg))
    assert jpeg.startswith(b"\xff\xd8") and header.pose is not None and len(header.pose) == 16
    assert header.intrinsics is not None and header.intrinsics["width"] == 64


def test_two_phones_stream_and_get_results(live_server: str, toy_frames: list) -> None:
    results = asyncio.run(simulate(live_server, 2, toy_frames, fps=20, duration=1.0))
    assert [s.device for s in results] == ["phone-1", "phone-2"]
    for stats in results:
        assert stats.sent >= 10 and stats.results >= stats.sent // 2
        assert stats.errors == 0 and min(stats.batch_sizes) >= 1
        assert "phone-" in stats.summary()


def test_summary_of_an_idle_phone() -> None:
    assert "sent 0" in PhoneStats("idle").summary()
