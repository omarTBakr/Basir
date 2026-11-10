"""Tests for basir/ingest/datasets/presets.py."""

from __future__ import annotations

import pytest

from basir.ingest.datasets.presets import PRESET_ALIASES, PRESETS, TUM_BASE_URL, get_preset_info, resolve_preset_key

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier, key",
    [
        ("fr1/room", "fr1/room"),
        ("FR1/Room", "fr1/room"),
        ("fr1_room", "fr1/room"),
        ("fr1-room", "fr1/room"),
        ("tum_fr1_room", "fr1/room"),
        ("freiburg1_room", "fr1/room"),
        ("rgbd_dataset_freiburg1_room", "fr1/room"),
        ("tum_fr2_desk", "fr2/desk"),
        ("fr3_long_office", "fr3/long_office"),
        ("rgbd_dataset_freiburg3_long_office_household", "fr3/long_office"),
        ("rgbd_dataset_freiburg1_360", "fr1/360"),
    ],
)
def test_preset_keys_and_aliases(identifier: str, key: str) -> None:
    assert resolve_preset_key(identifier) == key


@pytest.mark.parametrize("identifier", ["unknown_dataset_xyz", "", None])
def test_unknown_preset(identifier: str | None) -> None:
    assert resolve_preset_key(identifier) is None
    assert get_preset_info(identifier) is None


def test_presets_use_current_tum_host_and_match_names() -> None:
    for key, info in PRESETS.items():
        assert info["url"].startswith(TUM_BASE_URL + "/freiburg"), key
        assert info["url"].endswith(f"/{info['name']}.tgz"), key


def test_fr1_360_points_at_the_real_file() -> None:
    assert get_preset_info("fr1/360")["url"].endswith("/freiburg1/rgbd_dataset_freiburg1_360.tgz")


def test_every_alias_resolves_to_a_preset() -> None:
    assert set(PRESET_ALIASES.values()) == set(PRESETS)


@pytest.mark.network
@pytest.mark.parametrize("key", sorted(PRESETS))
def test_preset_url_is_live(key: str) -> None:
    import urllib.request

    request = urllib.request.Request(PRESETS[key]["url"], method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as res:
        assert res.status == 200
