"""Regression coverage for background recovery after failed MCP boot."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest  # ty: ignore[unresolved-import]


def test_cancelled_start_reaps_internal_transport_task() -> None:
    from tools import mcp_tool

    async def scenario() -> None:
        server = mcp_tool.MCPServerTask("meridian")

        async def blocked_run(self, config: dict) -> None:
            await asyncio.Event().wait()

        with patch.object(mcp_tool.MCPServerTask, "run", new=blocked_run):
            start_task = asyncio.create_task(server.start({"command": "blocked"}))
            await asyncio.sleep(0)
            internal_task = server._task
            assert internal_task is not None and not internal_task.done()
            start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)

        assert internal_task.done()
        assert server.session is None

    asyncio.run(scenario())


def test_cancelled_start_bounds_stubborn_transport_cleanup() -> None:
    from tools import mcp_tool

    async def scenario() -> None:
        server = mcp_tool.MCPServerTask("meridian")

        async def stubborn_run(self, config: dict) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.Event().wait()

        with patch.object(mcp_tool.MCPServerTask, "run", new=stubborn_run), patch.object(
            mcp_tool, "_TRANSPORT_CANCEL_TIMEOUT_SECONDS", 0.01
        ):
            start_task = asyncio.create_task(server.start({"command": "blocked"}))
            await asyncio.sleep(0)
            internal_task = server._task
            start_task.cancel()
            await asyncio.wait_for(
                asyncio.gather(start_task, return_exceptions=True), timeout=0.2
            )

        assert internal_task in mcp_tool._unreaped_transport_tasks
        internal_task.cancel()
        await asyncio.gather(internal_task, return_exceptions=True)

    asyncio.run(scenario())


def test_failed_initial_registration_retries_until_tools_are_registered() -> None:
    from tools import mcp_tool

    attempts = 0

    async def fake_discover(name: str, config: dict) -> list[str]:
        nonlocal attempts
        attempts += 1
        assert name == "meridian"
        assert config == {"url": "https://mcp.example.test/mcp"}
        if attempts == 1:
            raise ConnectionError("gateway still starting")
        return ["mcp_meridian_ping"]

    async def scenario() -> None:
        with patch.object(
            mcp_tool, "_discover_and_register_server", side_effect=fake_discover
        ), patch("tools.mcp_tool.asyncio.sleep", new_callable=AsyncMock):
            task = asyncio.create_task(
                mcp_tool._retry_failed_server_registration(
                    "meridian", {"url": "https://mcp.example.test/mcp"}
                )
            )
            with mcp_tool._lock:
                mcp_tool._background_connect_tasks["meridian"] = task
            await task

        assert attempts == 2
        assert "meridian" not in mcp_tool._background_connect_tasks

    asyncio.run(scenario())


def test_background_retry_uses_exponential_backoff_capped_at_five_minutes() -> None:
    from tools import mcp_tool

    delays: list[float] = []
    attempts = 0

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def fake_discover(name: str, config: dict) -> list[str]:
        nonlocal attempts
        attempts += 1
        if attempts < 11:
            raise ConnectionError("still unavailable")
        return ["mcp_meridian_ping"]

    async def scenario() -> None:
        with patch.object(
            mcp_tool, "_discover_and_register_server", side_effect=fake_discover
        ), patch("tools.mcp_tool.asyncio.sleep", side_effect=fake_sleep):
            await mcp_tool._retry_failed_server_registration(
                "meridian", {"url": "https://mcp.example.test/mcp"}
            )

    asyncio.run(scenario())
    assert delays == [1, 2, 4, 8, 16, 32, 64, 128, 256, 300, 300]


def test_scheduler_is_idempotent_and_skips_permanent_failures() -> None:
    from tools import mcp_tool

    async def scenario() -> None:
        sleeper = asyncio.Event()

        async def wait_forever(name: str, config: dict) -> None:
            await sleeper.wait()

        with patch.object(
            mcp_tool, "_retry_failed_server_registration", side_effect=wait_forever
        ):
            assert mcp_tool._schedule_failed_server_retry(
                "meridian", {"url": "https://mcp.example.test/mcp"}, ConnectionError()
            )
            assert not mcp_tool._schedule_failed_server_retry(
                "meridian", {"url": "https://mcp.example.test/mcp"}, ConnectionError()
            )
            task = mcp_tool._background_connect_tasks.pop("meridian")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert not mcp_tool._schedule_failed_server_retry(
            "bad-config", {"command": ""}, ValueError("missing command")
        )

    asyncio.run(scenario())


def test_ordinary_discovery_preserves_active_background_backoff() -> None:
    from tools import mcp_tool

    pending = MagicMock()
    pending.done.return_value = False
    with patch.object(mcp_tool, "_MCP_AVAILABLE", True), patch.object(
        mcp_tool, "_servers", {}
    ), patch.object(
        mcp_tool, "_background_connect_tasks", {"meridian": pending}
    ), patch.object(
        mcp_tool, "_existing_tool_names", return_value=[]
    ), patch.object(
        mcp_tool, "_connect_server", new_callable=AsyncMock
    ) as connect:
        assert mcp_tool.register_mcp_servers(
            {"meridian": {"url": "https://mcp.example.test/mcp"}}
        ) == []

    connect.assert_not_awaited()
    pending.cancel.assert_not_called()


def test_nested_permanent_failure_does_not_retry() -> None:
    from tools import mcp_tool

    grouped = ExceptionGroup(
        "connection failed",
        [ConnectionError("temporary"), ValueError("invalid configuration")],
    )
    assert not mcp_tool._should_retry_failed_initial_connection(grouped)


def test_implicit_exception_context_does_not_turn_transient_failure_permanent() -> None:
    from tools import mcp_tool

    try:
        try:
            raise PermissionError("expired credential")
        except PermissionError:
            raise ConnectionError("temporary disconnect")
    except ConnectionError as transient:
        assert isinstance(transient.__context__, PermissionError)
        with patch.object(
            mcp_tool, "_get_auth_error_types", return_value=(PermissionError,)
        ):
            assert not mcp_tool._is_auth_error(transient)
            assert mcp_tool._should_retry_failed_initial_connection(transient)


def test_registration_failure_shuts_down_unpublished_server() -> None:
    from tools import mcp_tool

    async def scenario() -> None:
        server = SimpleNamespace(
            _registered_tool_names=[],
            shutdown=AsyncMock(),
        )
        with patch.object(
            mcp_tool, "_connect_server", new=AsyncMock(return_value=server)
        ), patch.object(
            mcp_tool, "_register_server_tools", side_effect=RuntimeError("bad schema")
        ), patch.object(mcp_tool, "_servers", {}):
            with pytest.raises(RuntimeError, match="bad schema"):
                await mcp_tool._discover_and_register_server(
                    "meridian", {"url": "https://mcp.example.test/mcp"}
                )
            assert "meridian" not in mcp_tool._servers
        server.shutdown.assert_awaited_once()

    asyncio.run(scenario())


def test_live_agent_refreshes_recovered_tools_at_turn_boundary() -> None:
    from agent import conversation_loop
    from tools.registry import registry

    tool = {"type": "function", "function": {"name": "mcp_meridian_ping"}}
    agent = SimpleNamespace(
        _tool_registry_generation=registry._generation - 1,
        enabled_toolsets=None,
        disabled_toolsets=None,
        tools=[
            {"type": "function", "function": {"name": "old_registry"}},
            {"type": "function", "function": {"name": "memory_recall"}},
            {"type": "function", "function": {"name": "lcm_grep"}},
        ],
        _registry_tool_names={"old_registry"},
        valid_tool_names={"old_registry", "memory_recall", "lcm_grep"},
    )
    with patch("model_tools.get_tool_definitions", return_value=[tool]):
        notice = conversation_loop._refresh_tools_for_registry_change(agent)

    assert {item["function"]["name"] for item in agent.tools} == {
        "mcp_meridian_ping", "memory_recall", "lcm_grep"
    }
    assert agent.valid_tool_names == {
        "mcp_meridian_ping", "memory_recall", "lcm_grep"
    }
    assert agent._tool_registry_generation == registry._generation
    assert notice is not None
    assert "Added tools: mcp_meridian_ping" in notice


@pytest.mark.parametrize(
    "content",
    [
        "hello",
        [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
    ],
)
def test_tool_change_notice_is_ephemeral_current_turn_context(content) -> None:
    from agent.conversation_loop import _append_ephemeral_user_context

    original = list(content) if isinstance(content, list) else content
    result = _append_ephemeral_user_context(content, ["[tool registry changed]"])

    assert "tool registry changed" in str(result)
    assert content == original
    if isinstance(content, list):
        assert len(result) == len(content) + 1


def test_agent_init_snapshot_cannot_mask_concurrent_registry_change() -> None:
    from agent import agent_init
    from tools.registry import registry

    original_generation = registry._generation

    def mutate_during_resolution(**kwargs):
        registry._generation += 1
        return [{"type": "function", "function": {"name": "new_tool"}}]

    try:
        with patch.object(
            agent_init, "_ra", return_value=SimpleNamespace(
                get_tool_definitions=mutate_during_resolution
            )
        ):
            tools, captured_generation = agent_init._resolve_tool_snapshot(
                enabled_toolsets=None,
                disabled_toolsets=None,
                quiet_mode=True,
            )

        assert tools[0]["function"]["name"] == "new_tool"
        assert captured_generation == original_generation
        assert captured_generation != registry._generation
    finally:
        registry._generation = original_generation


def test_force_retry_reuses_server_published_before_cancel() -> None:
    from tools import mcp_tool

    published: Any = SimpleNamespace(_registered_tool_names=["mcp_meridian_ping"])
    old_loop = mcp_tool._mcp_loop
    old_thread = mcp_tool._mcp_thread
    try:
        mcp_tool._servers.clear()
        mcp_tool._background_connect_tasks.clear()
        mcp_tool._ensure_mcp_loop()

        async def install_retry() -> None:
            async def retry() -> None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    with mcp_tool._lock:
                        mcp_tool._servers["meridian"] = published  # ty: ignore[invalid-assignment]
                    raise

            task = asyncio.create_task(retry())
            with mcp_tool._lock:
                mcp_tool._background_connect_tasks["meridian"] = task

        mcp_tool._run_on_mcp_loop(install_retry)

        with patch.object(mcp_tool, "_MCP_AVAILABLE", True), patch.object(
            mcp_tool, "_connect_server", new_callable=AsyncMock
        ) as connect:
            names = mcp_tool.register_mcp_servers(
                {"meridian": {"url": "https://mcp.example.test/mcp"}},
                force_retry=True,
            )

        assert names == ["mcp_meridian_ping"]
        connect.assert_not_awaited()
    finally:
        mcp_tool._servers.clear()
        mcp_tool._background_connect_tasks.clear()
        mcp_tool._stop_mcp_loop()
        mcp_tool._mcp_loop = old_loop
        mcp_tool._mcp_thread = old_thread


def test_shutdown_gate_refuses_late_server_publication() -> None:
    from tools import mcp_tool

    async def scenario() -> None:
        server = SimpleNamespace(
            _registered_tool_names=[],
            shutdown=AsyncMock(),
        )
        old_flag = mcp_tool._mcp_shutting_down
        try:
            mcp_tool._mcp_shutting_down = True  # ty: ignore[invalid-assignment]
            with patch.object(
                mcp_tool, "_connect_server", new=AsyncMock(return_value=server)
            ), patch.object(mcp_tool, "_register_server_tools", return_value=[]), patch.object(
                mcp_tool, "_servers", {}
            ):
                with pytest.raises(RuntimeError, match="shutdown"):
                    await mcp_tool._discover_and_register_server(
                        "meridian", {"url": "https://mcp.example.test/mcp"}
                    )
            server.shutdown.assert_awaited_once()
        finally:
            mcp_tool._mcp_shutting_down = old_flag

    asyncio.run(scenario())


def test_foreground_discovery_is_rejected_while_shutdown_is_active() -> None:
    from tools import mcp_tool

    old_flag = mcp_tool._mcp_shutting_down
    try:
        mcp_tool._mcp_shutting_down = True  # ty: ignore[invalid-assignment]
        with patch.object(mcp_tool, "_MCP_AVAILABLE", True), patch.object(
            mcp_tool, "_connect_server", new_callable=AsyncMock
        ) as connect:
            names = mcp_tool.register_mcp_servers(
                {"meridian": {"url": "https://mcp.example.test/mcp"}}
            )

        assert names == mcp_tool._existing_tool_names()
        connect.assert_not_awaited()
    finally:
        mcp_tool._mcp_shutting_down = old_flag


def test_shutdown_resnapshots_unreaped_tasks_created_by_retry_cancellation() -> None:
    from tools import mcp_tool

    old_loop = mcp_tool._mcp_loop
    old_thread = mcp_tool._mcp_thread
    stale_loop = asyncio.new_event_loop()
    stale_task = stale_loop.create_task(asyncio.sleep(60))
    stale_loop.close()
    setattr(stale_task, "_log_destroy_pending", False)
    server: Any = SimpleNamespace(shutdown=AsyncMock())
    try:
        mcp_tool._servers.clear()
        mcp_tool._background_connect_tasks.clear()
        mcp_tool._unreaped_transport_tasks.clear()
        mcp_tool._ensure_mcp_loop()

        async def install_retry() -> None:
            async def retry() -> None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    mcp_tool._track_unreaped_transport_task(stale_task)
                    raise

            task = asyncio.create_task(retry())
            with mcp_tool._lock:
                mcp_tool._background_connect_tasks["meridian"] = task
                mcp_tool._servers["current"] = server  # ty: ignore[invalid-assignment]

        mcp_tool._run_on_mcp_loop(install_retry)
        mcp_tool.shutdown_mcp_servers()

        server.shutdown.assert_awaited_once()
        assert mcp_tool._servers == {}
        assert mcp_tool._background_connect_tasks == {}
        assert mcp_tool._unreaped_transport_tasks == set()
    finally:
        if mcp_tool._mcp_loop is not None:
            mcp_tool.shutdown_mcp_servers()
        mcp_tool._mcp_loop = old_loop
        mcp_tool._mcp_thread = old_thread


def test_shutdown_deadline_clears_state_for_cancellation_resistant_server() -> None:
    from tools import mcp_tool

    old_loop = mcp_tool._mcp_loop
    old_thread = mcp_tool._mcp_thread
    server: Any = SimpleNamespace(name="stubborn", shutdown=AsyncMock())

    async def never_finishes() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.Event().wait()

    server.shutdown.side_effect = never_finishes
    try:
        mcp_tool._servers.clear()
        mcp_tool._background_connect_tasks.clear()
        mcp_tool._unreaped_transport_tasks.clear()
        mcp_tool._ensure_mcp_loop()
        with mcp_tool._lock:
            mcp_tool._servers["stubborn"] = server  # ty: ignore[invalid-assignment]

        with patch.object(mcp_tool, "_TRANSPORT_CANCEL_TIMEOUT_SECONDS", 0.01):
            mcp_tool.shutdown_mcp_servers()

        assert mcp_tool._servers == {}
        assert mcp_tool._background_connect_tasks == {}
        assert mcp_tool._unreaped_transport_tasks == set()
        assert mcp_tool._mcp_loop is None
    finally:
        if mcp_tool._mcp_loop is not None:
            mcp_tool.shutdown_mcp_servers()
        mcp_tool._mcp_loop = old_loop
        mcp_tool._mcp_thread = old_thread


def test_shutdown_cancels_background_reconnect_when_no_server_connected() -> None:
    from tools import mcp_tool

    old_loop = mcp_tool._mcp_loop
    old_thread = mcp_tool._mcp_thread
    try:
        mcp_tool._servers.clear()
        mcp_tool._background_connect_tasks.clear()
        mcp_tool._ensure_mcp_loop()

        async def install_waiter() -> None:
            async def waiter() -> None:
                await asyncio.Event().wait()

            task = asyncio.create_task(waiter())
            with mcp_tool._lock:
                mcp_tool._background_connect_tasks["meridian"] = task

        mcp_tool._run_on_mcp_loop(install_waiter)
        mcp_tool.shutdown_mcp_servers()

        assert mcp_tool._background_connect_tasks == {}
        assert mcp_tool._mcp_loop is None
    finally:
        if mcp_tool._mcp_loop is not None:
            mcp_tool.shutdown_mcp_servers()
        mcp_tool._mcp_loop = old_loop
        mcp_tool._mcp_thread = old_thread
