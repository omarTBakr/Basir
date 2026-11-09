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
    (project / ".env").write_text("APP_NAME=FromProject\nAPI_PORT=1234\n")
    clean_env.setattr(config, "PROJECT_ROOT", project)

    s = Settings(_env_file=tmp_path / "non_existent.env")
    assert (s.app_name, s.api_port, s.gradio_port, s.debug) == ("Basir", 8000, 7860, True)


def test_project_env_is_used_by_default(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("APP_NAME=FromProject\n")
    clean_env.setattr(config, "PROJECT_ROOT", tmp_path)
    assert Settings().app_name == "FromProject"


def test_env_file_variable_is_used(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    custom = tmp_path / "custom.env"
    custom.write_text("APP_NAME=FromEnvFileVar\n")
    clean_env.setenv("ENV_FILE", str(custom))
    assert Settings().app_name == "FromEnvFileVar"


def test_loads_custom_env(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "APP_NAME=CustomBasir\nAPI_PORT=8888\nGRADIO_PORT=7777\n"
        "DEBUG=False\nDEVICE=cpu\nDATA_DIR=/tmp/basir_data\n"
    )
    s = Settings(_env_file=env_file)
    assert (s.app_name, s.api_port, s.gradio_port, s.debug, s.device) == ("CustomBasir", 8888, 7777, False, "cpu")
    assert s.datasets_dir == Path("/tmp/basir_data/datasets")


def test_environment_overrides_env_file(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "base.env"
    env_file.write_text("API_PORT=8000\nAPP_NAME=FromEnvFile\n")
    clean_env.setenv("API_PORT", "9999")
    s = Settings(_env_file=env_file)
    assert (s.api_port, s.app_name) == (9999, "FromEnvFile")


def test_basir_prefix_beats_plain_name(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("API_PORT", "9000")
    clean_env.setenv("BASIR_API_PORT", "9100")
    clean_env.setenv("BASIR_DEBUG", "False")
    s = Settings(_env_file=tmp_path / "empty.env")
    assert (s.api_port, s.debug) == (9100, False)


def test_keyword_arguments_win(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("BASIR_APP_NAME", "FromEnv")
    s = Settings(_env_file=tmp_path / "empty.env", app_name="ExplicitApp", api_port=9123)
    assert (s.app_name, s.api_port) == ("ExplicitApp", 9123)


def test_empty_values_mean_unset(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("QDRANT_API_KEY=\nLANGFUSE_BASE_URL=\nAPP_NAME=\n")
    s = Settings(_env_file=env_file)
    assert s.qdrant_api_key is None and s.langfuse_base_url is None
    assert s.app_name == "Basir"  # unset falls back to the default


def test_unknown_and_removed_keys_are_ignored(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    """Older .env files (QDRANT_HOST, body-logging keys) still load."""
    env_file = tmp_path / ".env"
    env_file.write_text("QDRANT_HOST=x\nQDRANT_PORT=1\nLOG_REQUEST_BODIES=True\nAPP_NAME=Old\n")
    s = Settings(_env_file=env_file)
    assert s.app_name == "Old"
    assert not hasattr(s, "qdrant_host")


def test_get_settings_is_cached_until_cleared(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("BASIR_APP_NAME", "First")
    first = get_settings()
    clean_env.setenv("BASIR_APP_NAME", "Second")
    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings().app_name == "Second"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

PROPERTIES = [
    "project_root", "base_data_dir", "datasets_dir", "weights_dir", "logs_dir", "qdrant_storage_dir",
    "configs_dir", "eval_datasets_dir", "log_file", "api_base_url", "api_ws_url", "gradio_url",
    "effective_device", "is_development", "is_production", "default_camera_intrinsics",
    "default_k_matrix", "keyframe_thresholds", "fusion_spatial_gate_m", "fusion_similarity_gate",
    "ranking_weights", "reacquire_frustum_margin", "default_target_nouns",
    "outputs_dir", "mlflow_uri", "mlflow_artifacts_dir", "langfuse_enabled",
]


@pytest.mark.parametrize("name", PROPERTIES)
def test_derived_values_are_properties_not_fields(name: str, tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=tmp_path / "e.env")
    assert name not in Settings.model_fields
    assert getattr(s, name) is not None


def test_urls_and_environment(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    s = Settings(
        _env_file=tmp_path / "e.env",
        api_host="127.0.0.1", api_port=8080, gradio_host="127.0.0.1", gradio_port=7861,
        ws_path="/ws/frames", app_env="production",
    )
    assert s.api_base_url == "http://127.0.0.1:8080"
    assert s.api_ws_url == "ws://127.0.0.1:8080/ws/frames"
    assert s.gradio_url == "http://127.0.0.1:7861"
    assert s.is_production and not s.is_development


def test_camera_intrinsics_and_k(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=tmp_path / "e.env")
    c = s.default_camera_intrinsics
    assert (c["fx"], c["fy"], c["cx"], c["cy"], c["depth_scale"]) == (525.0, 525.0, 319.5, 239.5, 5000.0)
    assert s.default_k_matrix == [[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]]


def test_design_constants(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=tmp_path / "e.env")
    assert s.fusion_spatial_gate_m == pytest.approx(0.5)  # PLAN.md §7
    assert s.keyframe_thresholds["min_interval_s"] == 1.0
    assert sum(s.ranking_weights.values()) == pytest.approx(1.0)
    assert {"laptop", "keys", "mug", "water bottle"} <= set(s.default_target_nouns)
    assert len(s.default_target_nouns) == len(set(s.default_target_nouns))


def test_data_dir_paths(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    relative = Settings(_env_file=tmp_path / "e.env", data_dir="data")
    assert relative.base_data_dir == (relative.project_root / "data").resolve()
    assert relative.datasets_dir == relative.base_data_dir / "datasets"
    assert Settings(_env_file=tmp_path / "e.env").base_data_dir == config.PROJECT_ROOT


def test_log_file(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=tmp_path / "e.env", data_dir=str(tmp_path)).log_file == tmp_path / "logs" / "basir.log"
    assert Settings(_env_file=tmp_path / "e.env", log_file_path="/var/log/b.log").log_file == Path("/var/log/b.log")


def test_explicit_device_is_normalised(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=tmp_path / "e.env", device="CPU ").effective_device == "cpu"


def test_observability_defaults(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=tmp_path / "e.env", data_dir=str(tmp_path))
    assert s.mlflow_uri == f"sqlite:///{tmp_path / 'mlflow.db'}"
    assert s.mlflow_artifacts_dir == tmp_path / "mlartifacts"
    assert s.outputs_dir == tmp_path / "outputs"
    assert s.langfuse_enabled is False
    assert Settings(_env_file=tmp_path / "e.env", mlflow_tracking_uri="http://mlflow:5000").mlflow_uri == "http://mlflow:5000"


def test_langfuse_needs_both_keys(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    assert not Settings(_env_file=tmp_path / "e.env", langfuse_public_key="pk").langfuse_enabled
    assert Settings(_env_file=tmp_path / "e.env", langfuse_public_key="pk", langfuse_secret_key="sk").langfuse_enabled
