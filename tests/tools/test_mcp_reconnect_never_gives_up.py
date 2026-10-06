"""The MCP client must not stop reconnecting, and must notice a GONE session.

Both defects were measured in production on 2026-08-18. An agent lost its
Meridian MCP server and stayed disconnected for three days across 59 failures,
recovering only when the container restarted.

Two causes, and they compound:

* ``retries`` was initialised once before the run loop and never reset -- not
  even on a successful connection -- so ``_MAX_RECONNECT_RETRIES = 5`` was a
  LIFETIME budget. Five blips spread over weeks killed the server for good.
* the 180s keepalive ran only ``if self.session:``, so the one state it could
  never detect was the session being absent, which is exactly what a failed
  reconnect leaves behind.
"""
import asyncio
import time

import pytest


async def _assert_inner_deadline(awaitable, *, watchdog_seconds=0.5):
    """A test watchdog cannot stand in for the production handshake timer."""
    handshake = asyncio.create_task(awaitable)
    try:
        completed, _ = await asyncio.wait({handshake}, timeout=watchdog_seconds)
        if not completed:
            pytest.fail("outer test watchdog fired; handshake has no deadline")
        with pytest.raises(TimeoutError):
            await handshake
    finally:
        if not handshake.done():
            handshake.cancel()
            await asyncio.gather(handshake, return_exceptions=True)


@pytest.mark.asyncio
async def test_stalled_reconnect_setup_expires_and_retries(monkeypatch):
    """A stalled initialize must not hold the reconnect loop forever."""
    from tools import mcp_tool

    attempts = 0
    cancelled = asyncio.Event()

    class HangingSession:
        async def initialize(self):
            nonlocal attempts
            attempts += 1
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    task = mcp_tool.MCPServerTask("test")
    task._ready.set()  # A previously connected server is reconnecting.
    real_sleep = asyncio.sleep

    async def no_wait(_delay):
        await real_sleep(0)

    async def connect(self, config):
        if attempts:
            self._shutdown_event.set()
            return
        await self._initialize_and_discover(HangingSession(), config)

    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_http", connect)
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_is_http", lambda self: True)
    monkeypatch.setattr(mcp_tool.asyncio, "sleep", no_wait)

    await asyncio.wait_for(
        task.run({"url": "https://example.invalid/mcp", "connect_timeout": 0.02}),
        timeout=0.5,
    )
    assert attempts == 1
    assert cancelled.is_set()
    assert task.session is None


@pytest.mark.asyncio
async def test_stalled_tool_discovery_expires_without_retaining_session(monkeypatch):
    """The handshake includes discovery, not only initialize()."""
    from tools import mcp_tool

    class Session:
        async def initialize(self):
            return object()

    async def stuck_discovery(self):
        await asyncio.Event().wait()

    task = mcp_tool.MCPServerTask("test")
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_discover_tools", stuck_discovery)

    await _assert_inner_deadline(
        task._initialize_and_discover(Session(), {"connect_timeout": 0.02})
    )
    assert task.session is None


@pytest.mark.asyncio
async def test_http_transport_entry_has_a_handshake_deadline(monkeypatch):
    """The SDK can hang before it yields the session to initialize()."""
    import httpx
    from tools import mcp_tool

    cancelled = asyncio.Event()

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class StalledTransport:
        async def __aenter__(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mcp_tool, "_MCP_HTTP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MCP_NEW_HTTP", True)
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setattr(
        mcp_tool, "streamable_http_client", lambda *args, **kwargs: StalledTransport(),
        raising=False,
    )
    task = mcp_tool.MCPServerTask("meridian")

    await _assert_inner_deadline(
        task._run_http({"url": "https://example.invalid/mcp", "connect_timeout": 0.02})
    )
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_sse_transport_entry_has_a_handshake_deadline(monkeypatch):
    """An SSE stream without an endpoint event must not stall reconnection."""
    from tools import mcp_tool

    cancelled = asyncio.Event()

    class StalledTransport:
        async def __aenter__(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mcp_tool, "_MCP_HTTP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "sse_client", lambda **kwargs: StalledTransport())
    task = mcp_tool.MCPServerTask("meridian")
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            task._run_http({
                "url": "https://example.invalid/sse", "transport": "sse",
                "connect_timeout": 0.02,
            }),
            timeout=0.5,
        )
    assert cancelled.is_set()
    assert time.monotonic() - started < 0.2


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "sse"])
async def test_successful_handshake_stays_connected_past_connect_timeout(monkeypatch, transport):
    """The handshake deadline must not kill a healthy long-lived session."""
    import httpx
    from tools import mcp_tool

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class Transport(Client):
        async def __aenter__(self):
            if transport == "sse":
                return object(), object()
            return object(), object(), lambda: "session"

    class Session(Client):
        async def initialize(self):
            return object()

        async def list_tools(self):
            return type("Tools", (), {"tools": []})()

    async def wait_for_shutdown(self):
        await asyncio.sleep(0.06)  # Longer than the 0.02s connect deadline.
        return "shutdown"

    monkeypatch.setattr(mcp_tool, "_MCP_HTTP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MCP_NEW_HTTP", True)
    monkeypatch.setattr(mcp_tool, "ClientSession", Session)
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_wait_for_lifecycle_event", wait_for_shutdown)
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setattr(mcp_tool, "streamable_http_client", lambda *args, **kwargs: Transport())
    monkeypatch.setattr(mcp_tool, "sse_client", lambda **kwargs: Transport())
    task = mcp_tool.MCPServerTask("meridian")
    config = {"url": "https://example.invalid/mcp", "connect_timeout": 0.02}
    if transport == "sse":
        config["transport"] = "sse"

    await asyncio.wait_for(task._run_http(config), timeout=0.5)
    assert task._ready.is_set()
    assert task.session is not None


@pytest.mark.asyncio
async def test_a_missing_session_triggers_a_reconnect(monkeypatch):
    """The health probe must cover the unhealthiest state.

    A session that is present and broken raised from list_tools and set the
    reconnect event. A session that is GONE hit `if self.session:` and was
    skipped, so nothing ever asked for it back.
    """
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_KEEPALIVE_INTERVAL", 0.01)
    task = mcp_tool.MCPServerTask("test")
    task.session = None

    reason = await asyncio.wait_for(task._wait_for_lifecycle_event(), timeout=5.0)
    assert reason == "reconnect"


@pytest.mark.asyncio
async def test_one_missing_interval_is_grace_not_a_reconnect(monkeypatch):
    """`self.session` is legitimately None for a moment during startup and
    during a reconnect. Firing on the first observation would fight the run
    loop instead of helping it."""
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_KEEPALIVE_INTERVAL", 0.05)
    task = mcp_tool.MCPServerTask("test")
    task.session = None

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(task._wait_for_lifecycle_event(), timeout=0.07)
    assert not task._reconnect_event.is_set()


@pytest.mark.asyncio
async def test_the_reconnect_budget_is_not_a_lifetime_cap(monkeypatch):
    """Six consecutive transport failures must not end the run loop.

    The old code returned after five, permanently, and only a process restart
    brought the server back.
    """
    from tools import mcp_tool

    attempts = 0

    async def always_fails(_self, _config):
        nonlocal attempts
        attempts += 1
        if attempts >= 8:
            raise asyncio.CancelledError
        raise ConnectionError("transport down")

    task = mcp_tool.MCPServerTask("test")
    task._ready.set()  # past the initial-connection path
    # MCPServerTask uses __slots__, so instance attributes are read-only.
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_http", always_fails, raising=False)
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_is_http", lambda _self: True, raising=False)
    # Capture the real sleep BEFORE patching, or the replacement calls itself.
    real_sleep = asyncio.sleep

    async def no_wait(_delay):
        await real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", no_wait)

    with pytest.raises(asyncio.CancelledError):
        await task.run({"url": "https://example.invalid/mcp"})

    assert attempts >= 6, (
        f"gave up after {attempts} attempts; a capped budget means one long "
        f"server redeploy disconnects the agent until the process restarts"
    )


@pytest.mark.asyncio
async def test_a_long_lived_connection_resets_the_failure_count(monkeypatch):
    """`retries` must count failures IN A ROW, not failures ever.

    The original bug was not only the ceiling. `retries` was initialised once
    before the loop and never reset -- not on a successful connection either --
    so five blips spread over weeks were indistinguishable from five in a row.
    An attempt that stayed up long enough to be a real session must clear it.
    """
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_RECONNECT_SUCCESS_SECONDS", 0.0)
    seen: list[int] = []
    attempts = 0

    async def up_then_down(_self, _config):
        nonlocal attempts
        attempts += 1
        if attempts >= 4:
            raise asyncio.CancelledError
        raise ConnectionError("dropped after a good run")

    real_sleep = asyncio.sleep

    async def no_wait(_delay):
        await real_sleep(0)

    def record(msg, *args, **kwargs):
        if "connection lost" in str(msg):
            seen.append(args[1])

    task = mcp_tool.MCPServerTask("test")
    task._ready.set()
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_http", up_then_down, raising=False)
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_is_http", lambda _self: True, raising=False)
    monkeypatch.setattr(mcp_tool.asyncio, "sleep", no_wait)
    monkeypatch.setattr(mcp_tool.logger, "warning", record)

    with pytest.raises(asyncio.CancelledError):
        await task.run({"url": "https://example.invalid/mcp"})

    # Every attempt "stayed up" past the success threshold, so each failure is
    # attempt 1 of a fresh sequence. Without the reset they climb 1, 2, 3.
    #
    # Assert the COUNT of log lines, not just their values. The warning is
    # throttled to `retries == 1 or retries % _RECONNECT_LOG_EVERY == 0`, so an
    # un-reset counter climbing 1, 2, 3 logs only ONCE and `set(seen) == {1}`
    # holds either way. scripts/sabotage.py caught that: removing the reset
    # left this test green until it checked how many lines appeared.
    assert len(seen) == 3, f"expected one line per failure, got {seen}"
    assert set(seen) == {1}, f"failure count did not reset: {seen}"


@pytest.mark.asyncio
async def test_removing_the_ceiling_did_not_remove_the_pacing(monkeypatch):
    """Unbounded retries must still back off, or the fix is a hot loop.

    A transport that fails INSTANTLY is the dangerous shape: with no ceiling
    and no growth, the run loop would spin as fast as the event loop allows
    against a server that is down. `backoff` is initialised once before the
    loop and only reset by a connection that stayed up, so the delays have to
    climb and then cap.

    The other tests patch sleep to a no-op, so none of them can see this.
    """
    from tools import mcp_tool

    delays: list[float] = []
    attempts = 0
    real_sleep = asyncio.sleep

    async def record(delay):
        delays.append(delay)
        await real_sleep(0)

    async def fails_instantly(_self, _config):
        nonlocal attempts
        attempts += 1
        if attempts >= 10:
            raise asyncio.CancelledError
        raise ConnectionError("refused")

    task = mcp_tool.MCPServerTask("test")
    task._ready.set()
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_http", fails_instantly, raising=False)
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_is_http", lambda _self: True, raising=False)
    monkeypatch.setattr(mcp_tool.asyncio, "sleep", record)

    with pytest.raises(asyncio.CancelledError):
        await task.run({"url": "https://example.invalid/mcp"})

    assert delays, "an instantly-failing transport must still sleep between attempts"

    # Pin the SCHEDULE, not just its shape. "sorted, grows, under the cap"
    # admits anything monotonic: a `backoff * 1.5` schedule gives
    # 1, 1.5, 2.25 ... 25.6, which is sorted, grows, stays under 60, and sums
    # past 30 -- every assertion satisfied while the cap is never reached and
    # the doubling is gone. Raised by CodeRabbit on this PR; verified by
    # working the 1.5x case by hand before taking the suggestion.
    cap = mcp_tool._MAX_BACKOFF_SECONDS
    expected: list[float] = []
    nxt = delays[0]
    for _ in delays:
        expected.append(nxt)
        nxt = min(nxt * 2, cap)
    assert delays == expected, f"unexpected backoff schedule: {delays}"
    assert delays[-1] == cap, f"backoff must actually reach its cap: {delays}"
