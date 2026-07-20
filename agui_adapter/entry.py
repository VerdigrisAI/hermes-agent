"""CLI entry point for the Hermes AG-UI adapter (HTTP/SSE server).

Usage::

    python -m agui_adapter          # or: hermes-agui
    PORT=8000 hermes-agui

Environment:
    PORT / HERMES_AGUI_PORT   listen port (default 8000)
    HERMES_AGUI_HOST          listen host (default 127.0.0.1)
    HERMES_AGUI_SESSION_TOKEN  required off-loopback; optional loopback defense-in-depth
    OPENAI_BASE_URL           LLM endpoint (aimock in tests)
    HERMES_AGUI_MODEL/PROVIDER/API_KEY/TOOLSETS   see session.AgentConfig
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet — happens during partial ``hermes update`` where git-reset landed
    # new code but ``uv pip install -e .`` didn't finish.  Missing bootstrap
    # means UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
    pass
else:
    # Stop a ``utils/``/``proxy/``/``ui/`` package in the launch directory from
    # shadowing Hermes's own modules — ``hermes agui`` can be started from any
    # cwd, including a project that has same-named packages on its path.
    hermes_bootstrap.harden_import_path()

# No ``from __future__ import annotations`` here: a future statement must
# precede every other import, which would displace hermes_bootstrap and break
# the entry-point contract (tests/test_hermes_bootstrap.py). It is unnecessary
# anyway — requires-python is >=3.11, so PEP 604/585 annotations are native.
# None of the other Hermes entry points carry one either.
import logging
import os
import sys


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    import uvicorn

    from agui_adapter.auth import require_token_or_refuse
    from agui_adapter.server import create_app

    host = os.environ.get("HERMES_AGUI_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT") or os.environ.get("HERMES_AGUI_PORT") or "8000")
    token = os.environ.get("HERMES_AGUI_SESSION_TOKEN") or None
    # main() is the authoritative fail-closed guard: it passes the SAME host to both
    # require_token_or_refuse and uvicorn.run below, so a network-accessible bind
    # without a usable token refuses to start. create_app() also re-checks against
    # the bound_host it is GIVEN, but that only protects an embedder that passes a
    # bound_host matching its real serve interface (see create_app's docstring).
    require_token_or_refuse(host, token)
    logging.getLogger(__name__).info("Starting Hermes AG-UI adapter on %s:%d", host, port)
    uvicorn.run(create_app(session_token=token, bound_host=host), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
