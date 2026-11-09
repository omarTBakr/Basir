"""Basir settings.

`Settings` holds two kinds of configuration:
    * Fields: values that can change per machine, read from `.env` and the environment.
    * Properties: derived paths and fixed design constants (thresholds, intrinsics, nouns).

Precedence, lowest to highest: field default < .env file < <FIELD> env var
< BASIR_<FIELD> env var < keyword arguments.

    from utilities.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from utilities.utility import parse_env_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Optional environment prefix: BASIR_API_PORT overrides API_PORT.
ENV_PREFIX = "basir_"


class Settings(BaseModel):
    """Application settings for Basir."""

    model_config = ConfigDict(extra="ignore")

    # --------------------------------------------------------------------------
    # Fields (set in .env or the environment)
    # --------------------------------------------------------------------------

    # Application
    app_name: str = Field(default="Basir", description="Application name")
    app_env: str = Field(default="development", description="development, staging or production")
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="DEBUG, INFO, WARNING or ERROR")

    # Compute
    device: str = Field(default="auto", description="auto, cuda, cpu or mps")

    # Server and UI
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")
    gradio_host: str = Field(default="0.0.0.0", description="Gradio UI host")
    gradio_port: int = Field(default=7860, description="Gradio UI port")
    ws_path: str = Field(default="/ws/stream", description="WebSocket path for the phone stream")

    # Vector database (Qdrant)
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    qdrant_collection_keyframes: str = Field(default="basir_keyframes", description="Keyframe embeddings")
    qdrant_collection_objects: str = Field(default="basir_objects", description="Object track embeddings")

    # Models
    detector_model: str = Field(default="IDEA-Research/grounding-dino-tiny", description="Open-vocabulary detector")
    segmenter_model: str = Field(default="facebook/sam2-hiera-tiny", description="Segmenter / video tracker (SAM 2)")
    embedder_model: str = Field(default="google/siglip-base-patch16-224", description="Text-image embedder (SigLIP)")
    dino_model: str = Field(default="facebook/dinov2-base", description="Image-image matching (DINOv2)")
    depth_model: str = Field(default="depth-anything/Depth-Anything-V2-metric-indoor-small", description="Metric mono-depth fallback")
    vlm_model: str = Field(default="microsoft/Florence-2-base", description="VLM for verification / captions")

    # External LLM / VLM API keys
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # Observability: Langfuse (runtime traces) and MLflow (evaluation runs)
    langfuse_public_key: Optional[str] = Field(default=None, description="Langfuse public key")
    langfuse_secret_key: Optional[str] = Field(default=None, description="Langfuse secret key")
    langfuse_base_url: Optional[str] = Field(default=None, description="Langfuse URL (default: Langfuse Cloud)")
    mlflow_tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking URI (default: SQLite in data_dir)")

    # Storage and logging
    data_dir: Optional[str] = Field(default=None, description="Data root (default: project root)")
    log_file_path: Optional[str] = Field(default=None, description="Log file (default: <logs_dir>/basir.log)")

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
    def weights_dir(self) -> Path:
        return self.base_data_dir / "weights"

    @property
    def logs_dir(self) -> Path:
        return self.base_data_dir / "logs"

    @property
    def qdrant_storage_dir(self) -> Path:
        return self.base_data_dir / "qdrant_storage"

    @property
    def configs_dir(self) -> Path:
        return PROJECT_ROOT / "configs"

    @property
    def eval_datasets_dir(self) -> Path:
        """Evaluation sessions and ground-truth labels."""
        return PROJECT_ROOT / "eval" / "datasets"

    @property
    def log_file(self) -> Path:
        if self.log_file_path and self.log_file_path.strip():
            path = Path(self.log_file_path.strip())
            return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        return self.logs_dir / "basir.log"

    @property
    def outputs_dir(self) -> Path:
        """Generated files such as point clouds (gitignored)."""
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
    # URLs and runtime
    # --------------------------------------------------------------------------

    @property
    def api_base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

    @property
    def api_ws_url(self) -> str:
        return f"ws://{self.api_host}:{self.api_port}{self.ws_path}"

    @property
    def gradio_url(self) -> str:
        return f"http://{self.gradio_host}:{self.gradio_port}"

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

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() in ("dev", "development", "local")

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in ("prod", "production")

    # --------------------------------------------------------------------------
    # Design constants (PLAN.md)
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

    @property
    def default_k_matrix(self) -> list[list[float]]:
        """3x3 intrinsic matrix K built from `default_camera_intrinsics`."""
        c = self.default_camera_intrinsics
        return [[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]]

    @property
    def keyframe_thresholds(self) -> dict[str, float]:
        """KeyframeSelector: keep a frame after this much time, motion or visual change."""
        return {
            "min_interval_s": 1.0,
            "min_translation_m": 0.30,
            "min_rotation_deg": 15.0,
            "min_visual_change": 0.15,
        }

    @property
    def fusion_spatial_gate_m(self) -> float:
        """Max distance for an observation to join an existing ObjectTrack (PLAN.md §7: ~0.5 m)."""
        return 0.50

    @property
    def fusion_similarity_gate(self) -> float:
        """Min cosine similarity of crop embeddings for the same ObjectTrack."""
        return 0.75

    @property
    def ranking_weights(self) -> dict[str, float]:
        """Query ranking (PLAN.md §7): w_sim·sim + w_vlm·verify + w_recency·recency − w_stale·stale."""
        return {"w_sim_text": 0.40, "w_vlm_verify": 0.30, "w_recency": 0.20, "w_stale": 0.10}

    @property
    def reacquire_frustum_margin(self) -> float:
        """Fraction of image size added around the frame when checking if an object is in view."""
        return 0.10

    @property
    def default_target_nouns(self) -> list[str]:
        """Household nouns for the always-on detection pass."""
        return [
            "laptop", "computer", "mouse", "keyboard", "monitor", "phone", "smartphone", "tablet",
            "keys", "keychain", "wallet", "badge", "glasses", "sunglasses",
            "cup", "mug", "water bottle", "thermos", "plate", "bowl", "fork", "spoon", "knife",
            "backpack", "handbag", "bag", "suitcase", "book", "notebook", "pen", "pencil",
            "remote control", "headphones", "charger", "power bank", "cable",
            "chair", "office chair", "armchair", "couch", "sofa", "desk", "table", "coffee table",
            "nightstand", "lamp", "trash can", "plant", "potted plant", "door", "window", "clock",
            "mirror", "cushion", "pillow", "jacket", "coat", "hat", "shoes", "slippers", "umbrella",
        ]


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
        if k.startswith(ENV_PREFIX) and k[len(ENV_PREFIX):] in fields:
            prefixed[k[len(ENV_PREFIX):]] = value
        elif k in fields:
            plain[k] = value
    return {**plain, **prefixed}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The shared Settings instance. Call `get_settings.cache_clear()` to reload."""
    return Settings()
