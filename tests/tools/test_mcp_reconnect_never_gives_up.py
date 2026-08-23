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

import pytest


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
