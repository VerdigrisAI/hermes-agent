"""Tests for ``${context:NAME}`` per-request interpolation in MCP-headers config.

Exercises both the resolution helpers (``_has_context_template``,
``_resolve_context_templates``, ``_split_static_and_templated_headers``)
and the per-request injection behavior via a real ``httpx.AsyncClient``
event hook — the latter is the contract that matters for downstream
consumers (Mercator's delegated-principal use case).
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

import httpx
import pytest

from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    register_session_context_var,
)
from tools.mcp_tool import (
    MCPServerTask,
    _apply_resolved_headers,
    _has_context_template,
    _resolve_context_templates,
    _resolve_frozen_headers,
    _split_static_and_templated_headers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def custom_var():
    """Register a fresh ContextVar each test under a stable name."""
    var: ContextVar = ContextVar("TEST_PRINCIPAL", default=_UNSET)
    register_session_context_var("TEST_PRINCIPAL", var)
    try:
        yield var
    finally:
        var.set(_UNSET)
        _VAR_MAP.pop("TEST_PRINCIPAL", None)


# ---------------------------------------------------------------------------
# Unit tests: detection + resolution helpers
# ---------------------------------------------------------------------------

class TestHasContextTemplate:
    """Detection helper — recognizes ``${context:NAME}`` in header values."""

    def test_detects_simple_template(self):
        assert _has_context_template("${context:FOO}")

    def test_detects_template_with_surrounding_text(self):
        assert _has_context_template("Bearer ${context:JWT}")

    def test_rejects_env_var_style(self):
        # Env vars are config-load-time, not per-request.
        assert not _has_context_template("${PLAIN_ENV}")

    def test_rejects_literal_string(self):
        assert not _has_context_template("plain-static-value")

    def test_rejects_non_string(self):
        assert not _has_context_template(None)
        assert not _has_context_template(42)
        assert not _has_context_template({"x": "${context:Y}"})


class TestResolveContextTemplates:
    """Resolution helper — substitutes ``${context:NAME}`` against session_context."""

    def test_resolves_set_value(self, custom_var):
        custom_var.set("alice@example.com")
        assert _resolve_context_templates("${context:TEST_PRINCIPAL}") == "alice@example.com"

    def test_unset_resolves_to_empty(self, custom_var):
        # ContextVar default is _UNSET — resolver falls back to os.environ
        # (also unset for TEST_PRINCIPAL) and finally returns "".
        assert _resolve_context_templates("${context:TEST_PRINCIPAL}") == ""

    def test_mixed_literal_and_template(self, custom_var):
        custom_var.set("opaque-token-xyz")
        assert (
            _resolve_context_templates("Bearer ${context:TEST_PRINCIPAL}")
            == "Bearer opaque-token-xyz"
        )

    def test_multiple_templates_in_one_string(self, custom_var):
        # Register a second var inline for this test.
        other: ContextVar = ContextVar("TEST_OTHER", default=_UNSET)
        register_session_context_var("TEST_OTHER", other)
        try:
            custom_var.set("a")
            other.set("b")
            result = _resolve_context_templates(
                "${context:TEST_PRINCIPAL}|${context:TEST_OTHER}"
            )
            assert result == "a|b"
        finally:
            other.set(_UNSET)
            _VAR_MAP.pop("TEST_OTHER", None)

    def test_non_string_passes_through(self):
        assert _resolve_context_templates(42) == 42  # type: ignore[arg-type]

    def test_unknown_name_resolves_to_empty(self):
        # No var registered for THIS_NAME — falls back to os.environ,
        # which is unset in tests, so empty string.
        assert _resolve_context_templates("${context:NEVER_REGISTERED}") == ""

    def test_whitespace_only_resolved_value_is_stripped_to_empty(self, custom_var):
        """A context-var holding only whitespace must NOT produce a
        header sent as ``X-Foo:    `` (HTTP-spec-violating, useless to
        the recipient).  The resolver strips leading/trailing whitespace
        so all three transports treat ``"   "`` the same as ``""`` —
        i.e. "drop this header" at the call site."""
        custom_var.set("   ")
        assert _resolve_context_templates("${context:TEST_PRINCIPAL}") == ""

    def test_leading_trailing_whitespace_stripped(self, custom_var):
        custom_var.set("  alice@example.com\t")
        assert _resolve_context_templates("${context:TEST_PRINCIPAL}") == "alice@example.com"

    def test_inner_whitespace_preserved(self, custom_var):
        """Strip only touches the outer edges — inner whitespace
        (e.g. ``"Bearer abc 123"``) is intentional and preserved."""
        custom_var.set("abc 123")
        assert (
            _resolve_context_templates("Bearer ${context:TEST_PRINCIPAL}")
            == "Bearer abc 123"
        )

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            (42, "42"),
            (0, "0"),
            (None, "None"),
            (False, "False"),
            (True, "True"),
            ([], "[]"),
            ({"k": "v"}, "{'k': 'v'}"),
        ],
    )
    def test_non_string_contextvar_value_coerces_to_str(self, raw_value, expected):
        """A plugin-registered ContextVar holding any non-string value must
        not crash the regex substitution. ``str()`` is the documented
        coercion path; future refactors that change this (e.g. drop-header-
        when-None) must update both this test and the docstring deliberately.
        """
        from contextvars import ContextVar as _ContextVar
        weird: _ContextVar = _ContextVar("WEIRD_VAL", default=_UNSET)
        register_session_context_var("WEIRD_VAL", weird)
        try:
            weird.set(raw_value)
            assert _resolve_context_templates("${context:WEIRD_VAL}") == expected
        finally:
            weird.set(_UNSET)
            _VAR_MAP.pop("WEIRD_VAL", None)


class TestResolveFrozenHeaders:
    """Freeze-time merge for SSE / legacy-HTTP transports — drops empty
    templated entries so an at-init resolution of an unset context var
    doesn't ship a blank header for the connection's lifetime.
    """

    def test_set_value_resolves_into_merged_dict(self, custom_var):
        custom_var.set("thomas@verdigris.co")
        merged = _resolve_frozen_headers(
            {"Authorization": "Bearer static"},
            {"X-Delegated-Principal": "${context:TEST_PRINCIPAL}"},
        )
        assert merged == {
            "Authorization": "Bearer static",
            "X-Delegated-Principal": "thomas@verdigris.co",
        }

    def test_empty_resolved_value_drops_header(self, custom_var):
        """The key fix: a templated header whose context var is unset
        at freeze time must be ABSENT from the result, not present with
        an empty value. Same drop-on-empty discipline as the HTTP
        per-request hook."""
        # custom_var default is _UNSET → resolves to ""
        merged = _resolve_frozen_headers(
            {"Authorization": "Bearer static"},
            {"X-Delegated-Principal": "${context:TEST_PRINCIPAL}"},
        )
        assert merged == {"Authorization": "Bearer static"}
        assert "X-Delegated-Principal" not in merged

    def test_whitespace_only_resolved_value_drops_header(self, custom_var):
        """Whitespace-only resolution is equivalent to empty — same drop
        behavior (avoids sending ``X-Foo:    `` which is HTTP-spec
        violating and useless to the recipient)."""
        custom_var.set("   \t")
        merged = _resolve_frozen_headers(
            {},
            {"X-Delegated-Principal": "${context:TEST_PRINCIPAL}"},
        )
        assert merged == {}

    def test_static_only(self):
        """Pure static headers pass through unchanged."""
        merged = _resolve_frozen_headers(
            {"Authorization": "Bearer x", "X-Trace-Id": "abc"},
            {},
        )
        assert merged == {"Authorization": "Bearer x", "X-Trace-Id": "abc"}

    def test_empty_input(self):
        assert _resolve_frozen_headers({}, {}) == {}


class TestSplitStaticAndTemplatedHeaders:
    """Header partition helper — separates client-default-safe from per-request."""

    def test_partitions_correctly(self):
        headers = {
            "Authorization": "Bearer static-token",
            "X-Delegated-Principal": "${context:PRINCIPAL_EMAIL}",
            "X-Other": "static-other",
            "X-Mixed": "prefix-${context:NAME}-suffix",
        }
        static, templated = _split_static_and_templated_headers(headers)
        assert static == {
            "Authorization": "Bearer static-token",
            "X-Other": "static-other",
        }
        assert templated == {
            "X-Delegated-Principal": "${context:PRINCIPAL_EMAIL}",
            "X-Mixed": "prefix-${context:NAME}-suffix",
        }

    def test_empty_input(self):
        assert _split_static_and_templated_headers({}) == ({}, {})

    def test_non_string_values_treated_as_static(self):
        # Edge case: a header value that's not a string (unusual but
        # possible from raw YAML) is treated as static — we don't try
        # to substitute on non-strings.
        headers = {"X-Number": 42, "X-Bool": True}
        static, templated = _split_static_and_templated_headers(headers)
        assert static == {"X-Number": 42, "X-Bool": True}
        assert templated == {}


# ---------------------------------------------------------------------------
# Z2O-1694: caller-side resolution + instance-attr bridge to the MCP loop
# ---------------------------------------------------------------------------
#
# The real MCP request runs on a dedicated background event loop (and a task
# spawned by the SDK's post_writer), so resolving ``${context:}`` inside the
# httpx request hook *there* always saw an empty context — the per-turn
# session vars are set on the caller's thread, never the MCP loop's. The fix
# resolves on the caller side and bridges the values across via an instance
# attribute (``MCPServerTask._outbound_resolved_headers``), applied under
# ``_rpc_lock`` so per-server calls can't race. These tests pin that contract.

class TestResolveTemplatedHeadersMethod:
    """Caller-side resolution: ``MCPServerTask._resolve_templated_headers``
    snapshots the current task's session context into concrete values."""

    def test_resolves_configured_templates(self, custom_var):
        server = MCPServerTask("meridian")
        server._templated_headers = {
            "X-Meridian-Delegated-Principal": "${context:TEST_PRINCIPAL}",
        }
        custom_var.set("thomas@verdigris.co")
        assert server._resolve_templated_headers() == {
            "X-Meridian-Delegated-Principal": "thomas@verdigris.co",
        }

    def test_unset_value_omitted_from_dict(self, custom_var):
        server = MCPServerTask("meridian")
        server._templated_headers = {
            "X-Meridian-Delegated-Principal": "${context:TEST_PRINCIPAL}",
        }
        # custom_var unset → "" → dropped (not present with empty value).
        assert server._resolve_templated_headers() == {}

    def test_no_templates_returns_empty(self):
        server = MCPServerTask("meridian")
        assert server._resolve_templated_headers() == {}


class TestApplyResolvedHeaders:
    """Loop-side application: ``_apply_resolved_headers`` stamps the already-
    resolved values onto the outbound request (no context lookup here)."""

    def test_injects_when_same_origin(self):
        req = httpx.Request("POST", "http://meridian.invalid/mcp")
        _apply_resolved_headers(
            req, same_origin=True,
            names=["X-Meridian-Delegated-Principal"],
            resolved={"X-Meridian-Delegated-Principal": "thomas@verdigris.co"},
        )
        assert req.headers["X-Meridian-Delegated-Principal"] == "thomas@verdigris.co"

    def test_drops_when_value_missing(self):
        req = httpx.Request("POST", "http://meridian.invalid/mcp")
        req.headers["X-Meridian-Delegated-Principal"] = "stale@example.com"
        _apply_resolved_headers(
            req, same_origin=True,
            names=["X-Meridian-Delegated-Principal"],
            resolved={},  # nothing resolved this call → header must be removed
        )
        assert "X-Meridian-Delegated-Principal" not in req.headers

    def test_strips_on_cross_origin(self):
        req = httpx.Request("POST", "http://attacker.invalid/harvest")
        req.headers["X-Meridian-Delegated-Principal"] = "thomas@verdigris.co"
        _apply_resolved_headers(
            req, same_origin=False,
            names=["X-Meridian-Delegated-Principal"],
            resolved={"X-Meridian-Delegated-Principal": "thomas@verdigris.co"},
        )
        assert "X-Meridian-Delegated-Principal" not in req.headers


def test_resolved_headers_survive_lost_caller_context(custom_var):
    """The crux of Z2O-1694: resolve while the per-turn var is set, then
    apply *after* that context is gone (simulating the MCP background loop,
    whose task never saw the var). The header must still be injected —
    proving the value travels via the instance attr, not a live context
    lookup at request time."""
    server = MCPServerTask("meridian")
    server._templated_headers = {
        "X-Meridian-Delegated-Principal": "${context:TEST_PRINCIPAL}",
    }
    custom_var.set("thomas@verdigris.co")
    resolved = server._resolve_templated_headers()        # caller thread/context
    server._outbound_resolved_headers = resolved          # bridged onto instance
    custom_var.set(_UNSET)                                 # caller context gone

    req = httpx.Request("POST", "http://meridian.invalid/mcp")
    _apply_resolved_headers(
        req, same_origin=True,
        names=server._templated_headers,
        resolved=server._outbound_resolved_headers,
    )
    assert req.headers["X-Meridian-Delegated-Principal"] == "thomas@verdigris.co"


# ---------------------------------------------------------------------------
# Integration tests: httpx request-event-hook mechanics
# ---------------------------------------------------------------------------
# NOTE: these exercise the httpx event-hook *mechanism* in the test's own
# event loop, where the calling task's context IS visible. The real MCP
# request runs on a dedicated background loop whose task can't see the
# caller's context (Z2O-1694), so the production hook no longer resolves
# ``${context:}`` here — it applies values resolved caller-side (see the
# TestResolveTemplatedHeadersMethod / _apply_resolved_headers tests above).
# These remain valid as httpx-behavior documentation.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_request_header_injection_via_event_hook(custom_var):
    """The pattern used inside ``_run_http`` — install a request event
    hook on httpx.AsyncClient that resolves ``${context:NAME}`` templates
    against the *calling task's* session context.  Verifies the per-request
    contract end-to-end (no MCP-stack mocking required)."""
    captured_headers: dict[str, str] = {}

    async def _capture_handler(request: httpx.Request) -> httpx.Response:
        captured_headers.clear()
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={"ok": True})

    templated = {"X-Delegated-Principal": "${context:TEST_PRINCIPAL}"}

    async def _inject(request: httpx.Request) -> None:
        for name, template in templated.items():
            resolved = _resolve_context_templates(template)
            if resolved:
                request.headers[name] = resolved
            elif name in request.headers:
                del request.headers[name]

    transport = httpx.MockTransport(_capture_handler)
    async with httpx.AsyncClient(
        transport=transport,
        event_hooks={"request": [_inject]},
        headers={"Authorization": "Bearer static-token"},
    ) as client:
        # Request 1: principal "alice" — both headers present.
        custom_var.set("alice@example.com")
        await client.get("http://example.invalid/mcp")
        assert captured_headers["x-delegated-principal"] == "alice@example.com"
        assert captured_headers["authorization"] == "Bearer static-token"

        # Request 2: principal "bob" — verifies per-request resolution
        # (the same client emits a different header value on the next call).
        custom_var.set("bob@example.com")
        await client.get("http://example.invalid/mcp")
        assert captured_headers["x-delegated-principal"] == "bob@example.com"

        # Request 3: principal cleared — header is dropped (NOT sent empty).
        custom_var.set("")
        await client.get("http://example.invalid/mcp")
        assert "x-delegated-principal" not in captured_headers


@pytest.mark.asyncio
async def test_principal_mutated_between_requests_in_same_task(custom_var):
    """Within a single task, mutating the ContextVar between requests must
    cause each request to carry the *current* value — not the value at
    hook-install time.

    Catches the latent bug class where the resolution would be cached at
    ``_run_http`` invocation time (e.g. ``resolved = _resolve_context_templates(template)``
    lifted out of the inner hook).  The
    ``test_asyncio_task_isolation_for_per_request_headers`` test below
    would silently miss this because each task only sets the var once.
    """
    seen: list[str] = []

    async def _capture(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-delegated-principal", "<missing>"))
        return httpx.Response(200)

    async def _inject(request: httpx.Request) -> None:
        resolved = _resolve_context_templates("${context:TEST_PRINCIPAL}")
        if resolved:
            request.headers["X-Delegated-Principal"] = resolved
        elif "x-delegated-principal" in request.headers:
            del request.headers["x-delegated-principal"]

    transport = httpx.MockTransport(_capture)
    async with httpx.AsyncClient(
        transport=transport, event_hooks={"request": [_inject]},
    ) as client:
        for value in ("alice@example.com", "bob@example.com", "carol@example.com"):
            custom_var.set(value)
            await client.get("http://example.invalid/mcp")

    assert seen == [
        "alice@example.com",
        "bob@example.com",
        "carol@example.com",
    ]


@pytest.mark.asyncio
async def test_templated_header_dropped_on_cross_origin_redirect(custom_var):
    """Identity-carrying templated headers must NOT follow a cross-origin
    redirect. Same threat model as ``Authorization``-stripping: if a
    redirect target is attacker-controlled, leaking ``X-Delegated-Principal``
    would let the attacker harvest user attribution claims.
    """
    # Two responses: the first redirects (302) to a different origin; the
    # second is captured so we can inspect the headers that actually crossed.
    captured_on_redirect_target: dict[str, str] = {}

    async def _handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "original.invalid":
            return httpx.Response(
                302,
                headers={"location": "http://attacker.invalid/harvest"},
            )
        captured_on_redirect_target.clear()
        captured_on_redirect_target.update(dict(request.headers))
        return httpx.Response(200)

    # Replicate the production response hook *and* request hook patterns,
    # including the origin guard inside the request hook (the response-side
    # strip alone is insufficient — the request hook fires again on the
    # redirected request).
    _original_url = httpx.URL("http://original.invalid/mcp")
    templated = {"X-Delegated-Principal": "${context:TEST_PRINCIPAL}"}

    async def _strip_identity_on_redirect(response: httpx.Response) -> None:
        if response.is_redirect and response.next_request:
            target = response.next_request.url
            if (target.scheme, target.host, target.port) != (
                _original_url.scheme, _original_url.host, _original_url.port,
            ):
                response.next_request.headers.pop("Authorization", None)
                response.next_request.headers.pop("authorization", None)
                for name in templated:
                    response.next_request.headers.pop(name, None)

    async def _inject(request: httpx.Request) -> None:
        target = request.url
        same_origin = (target.scheme, target.host, target.port) == (
            _original_url.scheme, _original_url.host, _original_url.port,
        )
        for name in templated:
            if not same_origin:
                request.headers.pop(name, None)
                continue
            resolved = _resolve_context_templates(templated[name])
            if resolved:
                request.headers[name] = resolved
            elif name in request.headers:
                del request.headers[name]

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(
        transport=transport,
        follow_redirects=True,
        event_hooks={
            "request": [_inject],
            "response": [_strip_identity_on_redirect],
        },
        headers={"Authorization": "Bearer static-token"},
    ) as client:
        custom_var.set("thomas@verdigris.co")
        await client.get("http://original.invalid/mcp")

    # The redirect target should have neither the static Authorization
    # header nor the templated delegated-principal header.
    assert "authorization" not in captured_on_redirect_target
    assert "x-delegated-principal" not in captured_on_redirect_target


@pytest.mark.asyncio
async def test_asyncio_task_isolation_for_per_request_headers(custom_var):
    """Two concurrent asyncio tasks setting different principals must each
    see their own value on outbound requests — no cross-task leak."""
    seen_per_task: dict[str, str] = {}

    async def _record_handler(request: httpx.Request) -> httpx.Response:
        # Pull the calling task's principal out of the request header.
        seen_per_task[request.url.path.lstrip("/")] = request.headers.get(
            "x-delegated-principal", "<missing>"
        )
        return httpx.Response(200)

    async def _inject(request: httpx.Request) -> None:
        resolved = _resolve_context_templates("${context:TEST_PRINCIPAL}")
        if resolved:
            request.headers["X-Delegated-Principal"] = resolved

    transport = httpx.MockTransport(_record_handler)
    async with httpx.AsyncClient(
        transport=transport, event_hooks={"request": [_inject]},
    ) as client:
        async def handler(path: str, principal: str, delay: float):
            custom_var.set(principal)
            await asyncio.sleep(delay)
            await client.get(f"http://example.invalid/{path}")

        await asyncio.gather(
            handler("alice-path", "alice@example.com", 0.02),
            handler("bob-path",   "bob@example.com",   0.01),
        )

    assert seen_per_task == {
        "alice-path": "alice@example.com",
        "bob-path":   "bob@example.com",
    }
