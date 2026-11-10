"""Tests for basir/ingest/datasets/layout.py (frame association)."""

from __future__ import annotations

from pathlib import Path

from basir.ingest.datasets.layout import associate_tum_dataset

# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


def test_associate_frames(tmp_path: Path) -> None:
    (tmp_path / "rgb.txt").write_text("# c\n1.000000 rgb/1.png\n1.033000 rgb/2.png\n1.066000 rgb/3.png\n")
    (tmp_path / "depth.txt").write_text("# c\n1.002000 d/1.png\n1.035000 d/2.png\n1.068000 d/3.png\n")
    (tmp_path / "groundtruth.txt").write_text(
        "# c\n1.000000 0.1 0.2 0.3 0 0 0 1\n1.033000 0.2 0.2 0.3 0 0 0 1\n1.066000 0.3 0.2 0.3 0 0 0 1\n"
    )
    assoc, assoc_gt = associate_tum_dataset(tmp_path, max_difference=0.02)

    rows = [line.split() for line in assoc.read_text().splitlines() if not line.startswith("#")]
    assert rows[0] == ["1.000000", "rgb/1.png", "1.002000", "d/1.png"]
    assert len(rows) == 3
    gt_rows = [line.split() for line in assoc_gt.read_text().splitlines() if not line.startswith("#")]
    assert len(gt_rows) == 3 and gt_rows[1][5] == "0.2"


def test_association_respects_max_difference(tmp_path: Path) -> None:
    (tmp_path / "rgb.txt").write_text("1.000 rgb/1.png\n2.000 rgb/2.png\n")
    (tmp_path / "depth.txt").write_text("1.010 d/1.png\n2.500 d/2.png\n")  # second is 0.5 s away
    assoc, assoc_gt = associate_tum_dataset(tmp_path, max_difference=0.02)
    rows = [line for line in assoc.read_text().splitlines() if not line.startswith("#")]
    assert len(rows) == 1
    assert assoc_gt is None  # no groundtruth.txt


def test_association_needs_rgb_and_depth(tmp_path: Path) -> None:
    assert associate_tum_dataset(tmp_path) == (None, None)
