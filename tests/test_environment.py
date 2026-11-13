"""Tests for utilities/environment.py."""

from __future__ import annotations

import pytest

import utilities.environment as environment


def test_package_versions_reports_missing_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment, "PACKAGES", ["numpy", "surely-not-installed-basir-pkg"])
    versions = environment.package_versions()
    assert versions["numpy"] and versions["surely-not-installed-basir-pkg"] is None
