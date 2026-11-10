"""Basir settings.

`Settings` holds two kinds of configuration:
    * Fields: values that can change per machine, read from `.env` and the environment.
    * Properties: derived paths and fixed constants.

Precedence, lowest to highest: field default < .env file < <FIELD> env var
< BASIR_<FIELD> env var < keyword arguments.

Only what the built phases use lives here; each later phase adds its own settings.

    from utilities.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from utilities.utility import parse_env_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Optional environment prefix: BASIR_DEVICE overrides DEVICE.
ENV_PREFIX = "basir_"


class Settings(BaseModel):
    """Application settings for Basir."""

    model_config = ConfigDict(extra="ignore")

    app_env: str = Field(default="development", description="development, staging or production")
    device: str = Field(default="auto", description="auto, cuda, cpu or mps")
    data_dir: str | None = Field(default=None, description="Data root (default: project root)")

    # Observability: MLflow (evaluation runs) and Langfuse (runtime traces)
    mlflow_tracking_uri: str | None = Field(default=None, description="Default: SQLite in the data root")
    langfuse_public_key: str | None = Field(default=None, description="Langfuse public key")
    langfuse_secret_key: str | None = Field(default=None, description="Langfuse secret key")
    langfuse_base_url: str | None = Field(default=None, description="Langfuse URL (default: Langfuse Cloud)")

    def __init__(self, _env_file: Path | str | None = None, **values: Any):
        """Load settings. An explicit `_env_file` is used on its own: if it does not
        exist, no .env file is read (it never falls back to the project's .env)."""
        fields = {name.lower(): name for name in type(self).model_fields}
        env_file = Path(_env_file) if _env_file else _default_env_file()

        data: dict[str, Any] = {}
        for source in (parse_env_file(env_file) if env_file else {}, _environment_values(fields)):
            for key, value in source.items():
                if (field := fields.get(key.lower())) is None:
                    continue
                if value.strip():
                    data[field] = value.strip()
                else:
                    data.pop(field, None)  # empty means "unset": fall back to the default
        data.update(values)
        super().__init__(**data)

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def base_data_dir(self) -> Path:
        """`data_dir` if set (relative paths are taken from the project root), else the project root."""
        if self.data_dir and self.data_dir.strip():
            path = Path(self.data_dir.strip())
            return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        return PROJECT_ROOT

    @property
    def datasets_dir(self) -> Path:
        """TUM RGB-D sequences and phone recordings."""
        return self.base_data_dir / "datasets"

    @property
    def outputs_dir(self) -> Path:
        """Generated files such as point clouds (git-ignored)."""
        return self.base_data_dir / "outputs"

    # --------------------------------------------------------------------------
    # Observability
    # --------------------------------------------------------------------------

    @property
    def mlflow_uri(self) -> str:
        """`mlflow_tracking_uri`, or a local SQLite store in the data directory."""
        if self.mlflow_tracking_uri and self.mlflow_tracking_uri.strip():
            return self.mlflow_tracking_uri.strip()
        return f"sqlite:///{self.base_data_dir / 'mlflow.db'}"

    @property
    def mlflow_artifacts_dir(self) -> Path:
        return self.base_data_dir / "mlartifacts"

    @property
    def langfuse_enabled(self) -> bool:
        """Tracing is on only when both Langfuse keys are set."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    @property
    def effective_device(self) -> str:
        """`device`, with 'auto' resolved to cuda, mps or cpu."""
        target = self.device.strip().lower()
        if target != "auto":
            return target
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # --------------------------------------------------------------------------
    # Constants
    # --------------------------------------------------------------------------

    @property
    def default_camera_intrinsics(self) -> dict[str, float]:
        """TUM RGB-D / Kinect v1 default calibration (640x480, depth factor 5000)."""
        return {
            "fx": 525.0,
            "fy": 525.0,
            "cx": 319.5,
            "cy": 239.5,
            "depth_scale": 5000.0,
            "width": 640.0,
            "height": 480.0,
        }


def _default_env_file() -> Path | None:
    """$ENV_FILE if set, else <project>/.env if it exists."""
    if os.getenv("ENV_FILE"):
        return Path(os.environ["ENV_FILE"])
    candidate = PROJECT_ROOT / ".env"
    return candidate if candidate.is_file() else None


def _environment_values(fields: dict[str, str]) -> dict[str, str]:
    """Environment variables that match a field; BASIR_<FIELD> wins over <FIELD>."""
    plain: dict[str, str] = {}
    prefixed: dict[str, str] = {}
    for key, value in os.environ.items():
        k = key.lower()
        if k.startswith(ENV_PREFIX) and k[len(ENV_PREFIX) :] in fields:
            prefixed[k[len(ENV_PREFIX) :]] = value
        elif k in fields:
            plain[k] = value
    return {**plain, **prefixed}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The shared Settings instance. Call `get_settings.cache_clear()` to reload."""
    return Settings()
