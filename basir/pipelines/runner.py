"""Run one pipeline step in Basir's environment: `python -m basir.pipelines.runner <pipeline> <task> <in> <out>`.

Airflow's tasks call this in a subprocess (see `basir.pipelines.airflow_task`), so Airflow's
own interpreter never imports torch, open3d or MLflow. `<in>` is the step's JSON payload,
`<out>` receives its JSON result. Exit codes: 0 done, EXIT_NO_RETRY for a failure that a
retry cannot fix (bad input, missing data, a failed gate), anything else for a failure worth
retrying. The error is printed to stderr, which ends up in the Airflow task log.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Any

EXIT_NO_RETRY = 3
ERROR_MARKER = "basir step failed:"  # run_pipeline.py looks for this line in the task log


def _phase0() -> tuple[Callable[[str, dict[str, Any]], Any], tuple[type[BaseException], ...]]:
    from basir.pipelines import phase0_steps

    return phase0_steps.execute, phase0_steps.NON_RETRYABLE


PIPELINES: dict[str, Callable[[], tuple[Callable[[str, dict[str, Any]], Any], tuple[type[BaseException], ...]]]] = {
    "phase0": _phase0,
}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 4 or args[0] not in PIPELINES:
        print(
            f"usage: python -m basir.pipelines.runner {{{','.join(PIPELINES)}}} TASK IN.json OUT.json", file=sys.stderr
        )
        return 2
    pipeline, task, source, target = args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    execute, non_retryable = PIPELINES[pipeline]()
    try:
        result = execute(task, json.loads(Path(source).read_text()))
    except non_retryable as e:
        traceback.print_exc()
        print(f"{ERROR_MARKER} {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_NO_RETRY
    except Exception as e:
        traceback.print_exc()
        print(f"{ERROR_MARKER} {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    Path(target).write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
