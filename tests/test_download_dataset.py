"""Tests for scripts/download_dataset.py (the command line)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.download_dataset import build_parser, main

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parser() -> None:
    args = build_parser().parse_args(["fr1/room", "--name", "room", "--dry-run"])
    assert (args.source, args.name, args.dry_run, args.output_dir) == ("fr1/room", "room", True, None)


def test_main_creates_toy_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--create-toy-dataset", "--name", "toy_room", "-o", str(tmp_path)]) == 0
    assert (tmp_path / "toy_room" / "intrinsics.json").is_file()
    assert "Dataset ready at" in capsys.readouterr().out


def test_main_list_presets(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-presets"]) == 0
    assert "fr1/room" in capsys.readouterr().out


def test_main_reports_failure(tmp_path: Path) -> None:
    assert main(["not_a_preset_or_path", "-o", str(tmp_path), "-q"]) == 1


def test_main_without_source_is_a_usage_error() -> None:
    with pytest.raises(SystemExit):
        main([])
