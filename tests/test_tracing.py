"""Tests for utilities/tracing.py (Langfuse, no-op without keys)."""

from __future__ import annotations

import socket

import pytest

import utilities.tracing as tracing
from utilities.tracing import flush, get_tracer, observe


@pytest.fixture(autouse=True)
def reset_tracer(clean_env: pytest.MonkeyPatch):
    get_tracer.cache_clear()
    yield clean_env
    get_tracer.cache_clear()


def test_without_keys_everything_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*_args, **_kwargs):
        raise AssertionError("tracing must not touch the network without keys")

    monkeypatch.setattr(socket, "create_connection", no_network)

    def add(a: int, b: int) -> int:
        return a + b

    traced = observe(name="add")(add)
    assert traced is add  # literally the same function
    assert traced(2, 3) == 5
    flush()
    assert get_tracer() is None


def test_with_keys_functions_are_wrapped(reset_tracer: pytest.MonkeyPatch) -> None:
    reset_tracer.setenv("BASIR_LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    reset_tracer.setenv("BASIR_LANGFUSE_SECRET_KEY", "sk-lf-test")
    reset_tracer.setenv("BASIR_LANGFUSE_BASE_URL", "http://127.0.0.1:9")  # nothing listens here
    reset_tracer.setenv("LANGFUSE_TRACING_ENABLED", "false")  # build the client, export nothing

    assert get_tracer() is not None

    def add(a: int, b: int) -> int:
        return a + b

    traced = observe(name="add")(add)
    assert traced is not add
    assert traced(2, 3) == 5


def test_tracer_start_failure_disables_tracing(
    reset_tracer: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_tracer.setenv("BASIR_LANGFUSE_PUBLIC_KEY", "pk")
    reset_tracer.setenv("BASIR_LANGFUSE_SECRET_KEY", "sk")

    import langfuse

    def broken(*_args, **_kwargs):
        raise RuntimeError("cannot start")

    monkeypatch.setattr(langfuse, "Langfuse", broken)
    assert get_tracer() is None
    assert observe()(len) is len


@pytest.mark.network
def test_live_langfuse_auth() -> None:
    """Runs only with -m network and real keys in .env."""
    from utilities.config import Settings

    if not Settings().langfuse_enabled:
        pytest.skip("no Langfuse keys configured")
    tracing.get_tracer.cache_clear()
    assert get_tracer().auth_check()
