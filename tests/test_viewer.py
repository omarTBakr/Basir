"""Tests for basir/api/viewer.py: a browser watching a phone gets its colour and depth side by side."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import numpy as np

from basir.api.phone_simulator import record
from basir.api.protocol import encode_frame
from basir.ingest.datasets.toy import create_toy_dataset
from basir.ingest.sources.tum import TUMDatasetSource


def test_viewer_gets_colour_depth_and_stats(client: TestClient, tmp_path: Path) -> None:
    frame = record(TUMDatasetSource(create_toy_dataset(tmp_path / "toy", num_frames=1)), depth_width=32)[0]
    with client.websocket_connect("/ws/stream/cam") as phone:
        phone.send_bytes(encode_frame(frame.header, frame.jpeg, frame.depth))
        phone.receive_text()
        with client.websocket_connect("/ws/view/cam") as viewer:
            viewer.send_text(json.dumps({"depth_max_m": 2.0}))
            colour, depth = viewer.receive_bytes(), viewer.receive_bytes()
            stats = json.loads(viewer.receive_text())
    assert colour[:1] == b"C" and colour[1:] == frame.jpeg
    assert depth[:1] == b"D"
    image = cv2.imdecode(np.frombuffer(depth[1:], np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (24, 32, 3)
    assert stats["connected"] and stats["pose"] and stats["colour"] == "64x48" and stats["depth"] == "32x24"
    assert stats["received"] == 1


def test_viewer_of_an_absent_phone_gets_offline_stats(client: TestClient) -> None:
    with client.websocket_connect("/ws/view/nobody") as viewer:
        stats = json.loads(viewer.receive_text())
    assert stats == {"type": "stats", "device": "nobody", "connected": False, "shown_fps": 0.0}
