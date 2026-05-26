"""Tests for the MCPServerTask reconnect signal.

When the OAuth layer cannot recover in-place (e.g., external refresh of a
single-use refresh_token made the SDK's in-memory refresh fail), the tool
handler signals MCPServerTask to tear down the current MCP session and
reconnect with fresh credentials. This file exercises the signal plumbing
in isolation from the full stdio/http transport machinery.
"""
import asyncio

import pytest


@pytest.mark.asyncio
async def test_reconnect_event_attribute_exists():
    """MCPServerTask has a _reconnect_event alongside _shutdown_event."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")
    assert hasattr(task, "_reconnect_event")
    assert isinstance(task._reconnect_event, asyncio.Event)
    assert not task._reconnect_event.is_set()


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_returns_reconnect():
    """When _reconnect_event fires, helper returns 'reconnect' and clears it."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._reconnect_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "reconnect"
    # Should have cleared so the next cycle starts fresh
    assert not task._reconnect_event.is_set()


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_returns_shutdown():
    """When _shutdown_event fires, helper returns 'shutdown'."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._shutdown_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "shutdown"


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_shutdown_wins_when_both_set():
    """If both events are set simultaneously, shutdown takes precedence."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._shutdown_event.set()
    task._reconnect_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "shutdown"


@pytest.mark.asyncio
async def test_successful_keepalive_schedules_tool_refresh():
    """After a successful keepalive, ``_schedule_tools_refresh`` is called so
    clients pick up tool-list changes from servers that don't emit
    ``notifications/tools/list_changed``.

    Without this, MCP gateways that proxy a backend registry (e.g. one
    that surfaces tools defined elsewhere) leave Hermes consumers with a
    stale tool list until process restart — the failure mode is silent
    (the agent reports "no such tool" truthfully) and hard to diagnose.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("test_keepalive")
    task.session = SimpleNamespace(list_tools=AsyncMock(return_value=None))

    # Drive exactly one keepalive cycle by patching asyncio.wait inside the
    # function's namespace. Iteration 1 returns an empty done-set (= timeout
    # = "no lifecycle event fired" = run keepalive). After the keepalive we
    # fire shutdown so iteration 2 exits cleanly.
    iteration = {"n": 0}
    real_wait = asyncio.wait

    async def fast_wait(aws, timeout=None, return_when=None):
        iteration["n"] += 1
        if iteration["n"] == 1:
            return set(), set(aws)
        return await real_wait(aws, timeout=0.01, return_when=return_when)

    with patch("tools.mcp_tool.asyncio.wait", side_effect=fast_wait), \
         patch.object(MCPServerTask, "_schedule_tools_refresh") as mock_schedule:
        # Set shutdown so iteration 2 returns immediately after we
        # come back through the loop.
        async def _set_shutdown_after_keepalive():
            # Wait a tick so we know iteration 1 has fired the keepalive
            # before we set shutdown for iteration 2.
            await asyncio.sleep(0.01)
            task._shutdown_event.set()
        asyncio.create_task(_set_shutdown_after_keepalive())
        reason = await task._wait_for_lifecycle_event()

    assert reason == "shutdown"
    assert mock_schedule.call_count >= 1, (
        "_schedule_tools_refresh must fire after each successful keepalive "
        "so tool-list changes from non-notifying servers are picked up."
    )


@pytest.mark.asyncio
async def test_failed_keepalive_does_not_schedule_refresh():
    """If the keepalive call itself fails, do NOT schedule a refresh — the
    reconnect path will rebuild the registry from scratch and scheduling a
    refresh against a broken session would surface a misleading error in
    the logs.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("test_keepalive_fail")
    task.session = SimpleNamespace(
        list_tools=AsyncMock(side_effect=RuntimeError("connection dead"))
    )

    iteration = {"n": 0}
    real_wait = asyncio.wait

    async def fast_wait(aws, timeout=None, return_when=None):
        iteration["n"] += 1
        if iteration["n"] == 1:
            return set(), set(aws)
        return await real_wait(aws, timeout=0.01, return_when=return_when)

    with patch("tools.mcp_tool.asyncio.wait", side_effect=fast_wait), \
         patch.object(MCPServerTask, "_schedule_tools_refresh") as mock_schedule:
        reason = await task._wait_for_lifecycle_event()

    # Keepalive failed → reconnect_event set → loop breaks → returns reconnect.
    assert reason == "reconnect"
    assert mock_schedule.call_count == 0, (
        "Must not schedule refresh when keepalive failed — the reconnect "
        "path will rebuild the registry from scratch."
    )
