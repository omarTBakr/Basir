"""Runtime tracing with Langfuse: what happened inside one request or query.

Code decorates its steps with `@observe`. When the Langfuse keys are not set it is a
no-op and nothing leaves the machine, so the same code runs with or without tracing.

    @observe(name="query.retrieve")
    def retrieve(text): ...

Input/output capture is off by default: Basir functions often receive frames and
images, which must not be uploaded. Turn it on per function with `capture=True`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import logging
from typing import Any, TypeVar, cast

from utilities.config import get_settings

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
    except Exception as e:  # tracing must never break the app
        logger.warning("Langfuse tracing disabled: %s", e)
        return None


def observe(name: str | None = None, *, as_type: str | None = None, capture: bool = False) -> Callable[[F], F]:
    """Trace a function as a Langfuse observation; a plain pass-through when tracing is off."""

    def decorator(func: F) -> F:
        if get_tracer() is None:
            return func
        from langfuse import observe as langfuse_observe

        # as_type is checked by Langfuse at runtime; its Literal type is not re-exported, hence the cast.
        decorate = langfuse_observe(
            name=name, as_type=cast(Any, as_type), capture_input=capture, capture_output=capture
        )
        return decorate(func)

    return decorator


def flush() -> None:
    """Send pending traces; call before a short-lived script exits."""
    tracer = get_tracer()
    if tracer is not None:
        try:
            tracer.flush()
        except Exception as e:
            logger.warning("Langfuse flush failed: %s", e)
