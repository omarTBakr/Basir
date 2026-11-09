"""Runtime tracing with Langfuse: what happened inside one request or query.

Code decorates its steps with `@observe` and wraps request handling in
`trace_context()`. When the Langfuse keys are not set, both are no-ops and nothing
leaves the machine, so the same code runs with or without tracing.

    @observe(name="query.retrieve")
    def retrieve(text): ...

    with trace_context(user_query="where is my mug?"):
        retrieve("mug")

Input/output capture is off by default: Basir functions often receive frames and
images, which must not be uploaded. Turn it on per function with `capture=True`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from functools import lru_cache
import logging
from typing import Any, TypeVar

from utilities.config import get_settings
from utilities.middleware import get_request_id

logger = logging.getLogger("basir.tracing")

F = TypeVar("F", bound=Callable[..., Any])


@lru_cache(maxsize=1)
def get_tracer() -> Any | None:
    """The Langfuse client, or None when tracing is not configured or fails to start."""
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url or None,
            environment=settings.app_env,
        )
    except Exception as e:  # noqa: BLE001 - tracing must never break the app
        logger.warning("Langfuse tracing disabled: %s", e)
        return None


def observe(name: str | None = None, *, as_type: str | None = None, capture: bool = False) -> Callable[[F], F]:
    """Trace a function as a Langfuse observation; a plain pass-through when tracing is off."""

    def decorator(func: F) -> F:
        if get_tracer() is None:
            return func
        from langfuse import observe as langfuse_observe

        return langfuse_observe(
            func, name=name, as_type=as_type, capture_input=capture, capture_output=capture
        )

    return decorator


@contextmanager
def trace_context(**metadata: Any) -> Iterator[None]:
    """Group everything traced inside under the current request ID (from the middleware)."""
    if get_tracer() is None:
        with nullcontext():
            yield
        return
    from langfuse import propagate_attributes

    request_id = get_request_id()
    with propagate_attributes(
        session_id=request_id,
        metadata={k: str(v) for k, v in metadata.items()} | ({"request_id": request_id} if request_id else {}),
    ):
        yield


def flush() -> None:
    """Send pending traces; call before a short-lived script exits."""
    tracer = get_tracer()
    if tracer is not None:
        try:
            tracer.flush()
        except Exception as e:  # noqa: BLE001
            logger.warning("Langfuse flush failed: %s", e)
