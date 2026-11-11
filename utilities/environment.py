"""What this machine has installed: the Python packages Basir depends on and their versions."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["torch", "numpy", "opencv-python", "open3d", "pydantic", "tqdm", "mlflow", "langfuse", "temporalio"]


def package_versions() -> dict[str, str | None]:
    """Installed version of each package in PACKAGES (None if missing)."""
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions
