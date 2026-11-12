"""The Airflow side of a pipeline step: run it with Basir's interpreter and hand back its JSON result.

Airflow's image has two Python environments: Airflow's own, which parses `dags/` and runs the
task wrappers, and Basir's (`BASIR_PYTHON`, built from uv.lock), which has torch, open3d and
MLflow. Each wrapper calls `run_step`, which starts `python -m basir.pipelines.runner` in
Basir's environment; its output streams into the Airflow task log.

No Airflow imports here, so the module is testable in the project venv; the DAG file turns
`StepFailed` into `AirflowFailException` (fail without retry).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from basir.pipelines.runner import EXIT_NO_RETRY

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class StepFailed(RuntimeError):
    """The step failed in a way a retry cannot fix."""


class StepCrashed(RuntimeError):
    """The step failed in a way a retry might fix (a network error, a killed process)."""


def basir_python() -> str:
    """Basir's interpreter: $BASIR_PYTHON (set in the Airflow image), else the current one."""
    return os.environ.get("BASIR_PYTHON") or sys.executable


def run_step(pipeline: str, task: str, payload: dict[str, Any], timeout: float | None = None) -> Any:
    """Run `task` of `pipeline` with `payload` in Basir's environment; returns the step's result."""
    with tempfile.TemporaryDirectory(prefix=f"basir-{task}-") as tmp:
        source, target = Path(tmp) / "in.json", Path(tmp) / "out.json"
        source.write_text(json.dumps(payload))
        env = os.environ | {
            "PYTHONPATH": os.pathsep.join(filter(None, [str(PROJECT_ROOT), os.environ.get("PYTHONPATH")]))
        }
        command = [basir_python(), "-m", "basir.pipelines.runner", pipeline, task, str(source), str(target)]
        code = subprocess.run(command, env=env, cwd=PROJECT_ROOT, timeout=timeout, check=False).returncode
        if code == EXIT_NO_RETRY:
            raise StepFailed(f"{pipeline}.{task} failed; see the error above")
        if code != 0:
            raise StepCrashed(f"{pipeline}.{task} exited with code {code}; see the error above")
        return json.loads(target.read_text())
