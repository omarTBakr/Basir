"""Tests for utilities/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import utilities.config as config
from utilities.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Loading and precedence
# ---------------------------------------------------------------------------


def test_defaults_when_explicit_file_is_missing(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    """A missing explicit env file must NOT fall back to the project's real .env."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("APP_ENV=production\nDEVICE=cpu\n")
    clean_env.setattr(config, "PROJECT_ROOT", project)

    s = Settings(_env_file=tmp_path / "non_existent.env")
    assert (s.app_env, s.device, s.data_dir) == ("development", "auto", None)
    assert s.prometheus_url == "http://127.0.0.1:9090"


def test_project_env_is_used_by_default(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("APP_ENV=staging\n")
    clean_env.setattr(config, "PROJECT_ROOT", tmp_path)
    assert Settings().app_env == "staging"


def test_env_file_variable_is_used(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    custom = tmp_path / "custom.env"
    custom.write_text("APP_ENV=from-env-file-var\n")
    clean_env.setenv("ENV_FILE", str(custom))
    assert Settings().app_env == "from-env-file-var"


def test_loads_custom_env(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("APP_ENV=production\nDEVICE=cpu\nDATA_DIR=/tmp/basir_data\nMLFLOW_TRACKING_URI=http://m:5000\n")
    s = Settings(_env_file=env_file)
    assert (s.app_env, s.device, s.mlflow_uri) == ("production", "cpu", "http://m:5000")
    assert s.datasets_dir == Path("/tmp/basir_data/datasets")


def test_environment_overrides_env_file(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "base.env"
    env_file.write_text("DEVICE=cpu\nAPP_ENV=from-file\n")
    clean_env.setenv("DEVICE", "cuda")
    s = Settings(_env_file=env_file)
    assert (s.device, s.app_env) == ("cuda", "from-file")


def test_basir_prefix_beats_plain_name(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("DEVICE", "cuda")
    clean_env.setenv("BASIR_DEVICE", "cpu")
    clean_env.setenv("BASIR_APP_ENV", "staging")
    s = Settings(_env_file=tmp_path / "empty.env")
    assert (s.device, s.app_env) == ("cpu", "staging")


def test_keyword_arguments_win(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("BASIR_DEVICE", "cuda")
    s = Settings(_env_file=tmp_path / "empty.env", device="cpu", app_env="production")
    assert (s.device, s.app_env) == ("cpu", "production")


def test_empty_values_mean_unset(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LANGFUSE_BASE_URL=\nDATA_DIR=\nAPP_ENV=\n")
    s = Settings(_env_file=env_file)
    assert s.langfuse_base_url is None and s.data_dir is None
    assert s.app_env == "development"  # unset falls back to the default


def test_unknown_and_removed_keys_are_ignored(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    """Older .env files (Qdrant, model IDs, server settings) still load."""
    env_file = tmp_path / ".env"
    env_file.write_text("QDRANT_URL=http://x\nAPI_PORT=1\nDETECTOR_MODEL=y\nAPP_ENV=old\n")
    s = Settings(_env_file=env_file)
    assert s.app_env == "old"
    assert not hasattr(s, "qdrant_url")


def test_get_settings_is_cached_until_cleared(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("BASIR_APP_ENV", "first")
    first = get_settings()
    clean_env.setenv("BASIR_APP_ENV", "second")
    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings().app_env == "second"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

PROPERTIES = [
    "project_root",
    "base_data_dir",
    "datasets_dir",
    "outputs_dir",
    "mlflow_uri",
    "mlflow_artifacts_dir",
    "langfuse_enabled",
    "effective_device",
    "default_camera_intrinsics",
]


@pytest.mark.parametrize("name", PROPERTIES)
def test_derived_values_are_properties_not_fields(name: str, tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=tmp_path / "e.env")
    assert name not in Settings.model_fields
    assert getattr(s, name) is not None


def test_data_dir_paths(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    relative = Settings(_env_file=tmp_path / "e.env", data_dir="data")
    assert relative.base_data_dir == (relative.project_root / "data").resolve()
    assert relative.datasets_dir == relative.base_data_dir / "datasets"
    assert relative.outputs_dir == relative.base_data_dir / "outputs"
    assert Settings(_env_file=tmp_path / "e.env").base_data_dir == config.PROJECT_ROOT


def test_camera_intrinsics(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    c = Settings(_env_file=tmp_path / "e.env").default_camera_intrinsics
    assert (c["fx"], c["fy"], c["cx"], c["cy"], c["depth_scale"]) == (525.0, 525.0, 319.5, 239.5, 5000.0)
    assert (c["width"], c["height"]) == (640.0, 480.0)


def test_explicit_device_is_normalised(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=tmp_path / "e.env", device="CPU ").effective_device == "cpu"


def test_observability_defaults(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=tmp_path / "e.env", data_dir=str(tmp_path))
    assert s.mlflow_uri == f"sqlite:///{tmp_path / 'mlflow.db'}"
    assert s.mlflow_artifacts_dir == tmp_path / "mlartifacts"
    assert s.langfuse_enabled is False


def test_langfuse_needs_both_keys(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    assert not Settings(_env_file=tmp_path / "e.env", langfuse_public_key="pk").langfuse_enabled
    assert Settings(_env_file=tmp_path / "e.env", langfuse_public_key="pk", langfuse_secret_key="sk").langfuse_enabled


def test_temporal_address_default_and_override(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=tmp_path / "e.env").temporal_address == "127.0.0.1:7233"
    clean_env.setenv("BASIR_TEMPORAL_ADDRESS", "temporal:7233")
    assert Settings(_env_file=tmp_path / "e.env").temporal_address == "temporal:7233"
