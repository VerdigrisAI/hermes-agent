"""Tests for /background gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

import asyncio
import json
import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource


def _make_event(text="/background", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner with minimal mocks."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()

    mock_store = MagicMock()
    runner.session_store = mock_store

    from gateway.hooks import HookRegistry
    runner.hooks = HookRegistry()

    return runner


# ---------------------------------------------------------------------------
# _handle_background_command
# ---------------------------------------------------------------------------


class TestHandleBackgroundCommand:
    """Tests for GatewayRunner._handle_background_command."""

    @pytest.mark.asyncio
    async def test_no_prompt_shows_usage(self):
        """Running /background with no prompt shows usage."""
        runner = _make_runner()
        event = _make_event(text="/background")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result
        assert "/background" in result

    @pytest.mark.asyncio
    async def test_bg_alias_no_prompt_shows_usage(self):
        """Running /bg with no prompt shows usage."""
        runner = _make_runner()
        event = _make_event(text="/bg")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_empty_prompt_shows_usage(self):
        """Running /background with only whitespace shows usage."""
        runner = _make_runner()
        event = _make_event(text="/background   ")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_valid_prompt_starts_task(self):
        """Running /background with a prompt returns confirmation and starts task."""
        runner = _make_runner()

        # Patch asyncio.create_task to capture the coroutine
        created_tasks = []
        original_create_task = asyncio.create_task

        def capture_task(coro, *args, **kwargs):
            # Close the coroutine to avoid warnings
            coro.close()
            mock_task = MagicMock()
            created_tasks.append(mock_task)
            return mock_task

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            event = _make_event(text="/background Summarize the top HN stories")
            result = await runner._handle_background_command(event)

        assert "🔄" in result
        assert "Background task started" in result
        assert "bg_" in result  # task ID starts with bg_
        assert "Summarize the top HN stories" in result
        assert len(created_tasks) == 1  # background task was created

    @pytest.mark.asyncio
    async def test_telegram_dm_topic_passes_trigger_anchor_to_task(self):
        """Telegram private-topic completion sends need the original command message id."""
        runner = _make_runner()
        runner._run_background_task = AsyncMock()

        def capture_task(coro, *args, **kwargs):
            coro.close()
            mock_task = MagicMock()
            return mock_task

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            chat_type="dm",
            thread_id="20197",
        )
        event = MessageEvent(
            text="/background summarize",
            source=source,
            message_id="463",
            reply_to_message_id="462",
        )

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            result = await runner._handle_background_command(event)

        assert "Background task started" in result
        runner._run_background_task.assert_called_once()
        assert runner._run_background_task.call_args.kwargs["event_message_id"] == "463"

    @pytest.mark.asyncio
    async def test_slack_slash_background_captures_private_reply_owner(self):
        runner = _make_runner()
        adapter = MagicMock()
        adapter.private_reply_user_id.return_value = "U_PRIVATE"
        runner.adapters[Platform.SLACK] = adapter
        runner._run_background_task = AsyncMock()

        def capture_task(coro, *args, **kwargs):
            coro.close()
            return MagicMock()

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            event = _make_event(
                text="/background private work",
                platform=Platform.SLACK,
                user_id="U_PRIVATE",
                chat_id="C1",
            )
            await runner._handle_background_command(event)

        job_state = runner._run_background_task.call_args.kwargs["job_state"]
        assert job_state["private_user_id"] == "U_PRIVATE"

    @pytest.mark.asyncio
    async def test_prompt_truncated_in_preview(self):
        """Long prompts are truncated to 60 chars in the confirmation message."""
        runner = _make_runner()
        long_prompt = "A" * 100

        with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
            event = _make_event(text=f"/background {long_prompt}")
            result = await runner._handle_background_command(event)

        assert "..." in result
        # Should not contain the full prompt
        assert long_prompt not in result

    @pytest.mark.asyncio
    async def test_task_id_is_unique(self):
        """Each background task gets a unique task ID."""
        runner = _make_runner()
        task_ids = set()

        with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
            for i in range(5):
                event = _make_event(text=f"/background task {i}")
                result = await runner._handle_background_command(event)
                # Extract task ID from result (format: "Task ID: bg_HHMMSS_hex")
                for line in result.split("\n"):
                    if "Task ID:" in line:
                        tid = line.split("Task ID:")[1].strip()
                        task_ids.add(tid)

        assert len(task_ids) == 5  # all unique

    @pytest.mark.asyncio
    async def test_works_across_platforms(self):
        """The /background command works for all platforms."""
        for platform in [Platform.TELEGRAM, Platform.DISCORD, Platform.SLACK]:
            runner = _make_runner()
            with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
                event = _make_event(
                    text="/background test task",
                    platform=platform,
                )
                result = await runner._handle_background_command(event)
                assert "Background task started" in result


# ---------------------------------------------------------------------------
# _run_background_task
# ---------------------------------------------------------------------------


class TestRunBackgroundTask:
    @pytest.mark.asyncio
    async def test_missing_adapter_records_terminal_outcome(self):
        runner = _make_runner()
        runner.adapters = {}
        runner._background_task_outcomes = {}
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")

        await runner._run_background_task("test", source, "bg_no_adapter")

        assert (
            runner._background_task_outcomes["bg_no_adapter"]["status"]
            == "failed_and_undelivered"
        )

    """Tests for GatewayRunner._run_background_task (the actual execution)."""

    @pytest.mark.asyncio
    async def test_no_adapter_returns_silently(self):
        """When no adapter is available, the task returns without error."""
        runner = _make_runner()
        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )
        # No adapters set — should not raise
        await runner._run_background_task("test prompt", source, "bg_test")

    @pytest.mark.asyncio
    async def test_no_credentials_sends_error(self):
        """When provider credentials are missing, an error is sent."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter._send_with_retry = AsyncMock(
            return_value=SendResult(success=True, message_id="failure-1")
        )
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": None}):
            await runner._run_background_task("test prompt", source, "bg_test")

        # Should have sent an error message
        mock_adapter._send_with_retry.assert_awaited_once()
        call_args = mock_adapter._send_with_retry.call_args
        assert "failed" in call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "").lower()

    @pytest.mark.asyncio
    async def test_no_credentials_preserves_delivered_failure_notice(self):
        runner = _make_runner()
        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock(
            return_value=SendResult(
                success=False,
                error="delivery exhausted",
                failure_notice_delivered=True,
            )
        )
        runner.adapters[Platform.SLACK] = adapter
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": None}):
            await runner._run_background_task("work", source, "bg_credentials")

        assert (
            runner._background_task_outcomes["bg_credentials"]["status"]
            == "delivery_failed_notified"
        )

    @pytest.mark.asyncio
    async def test_successful_task_sends_result(self):
        """When the agent completes successfully, the result is sent."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter._send_with_retry = AsyncMock(
            return_value=SendResult(success=True, message_id="completion-1")
        )
        mock_adapter.extract_media = MagicMock(return_value=([], "Hello from background!"))
        mock_adapter.extract_images = MagicMock(return_value=([], "Hello from background!"))
        mock_adapter.extract_local_files = MagicMock(return_value=([], "Hello from background!"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        mock_result = {"final_response": "Hello from background!", "messages": []}
        job_state = {
            "source": source,
            "agent": None,
            "thread_started": threading.Event(),
            "thread_done": threading.Event(),
        }

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.shutdown_memory_provider = MagicMock()
            mock_agent_instance.close = MagicMock()
            mock_agent_instance.run_conversation.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await runner._run_background_task(
                "say hello",
                source,
                "bg_test",
                job_state=job_state,
            )

        # Should have sent the result
        mock_adapter._send_with_retry.assert_awaited_once()
        call_args = mock_adapter._send_with_retry.call_args
        content = call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "Background task result" in content
        assert "Hello from background!" in content
        assert runner._background_task_outcomes["bg_test"]["status"] == "success"
        assert job_state["agent"] is mock_agent_instance
        assert job_state["thread_started"].is_set()
        assert job_state["thread_done"].is_set()
        mock_agent_instance.shutdown_memory_provider.assert_called_once()
        mock_agent_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_private_background_completion_never_uses_public_send(self):
        runner = _make_runner()
        adapter = MagicMock()
        adapter.send_private_notice = AsyncMock(return_value=SendResult(success=True))
        adapter._send_with_retry = AsyncMock()
        adapter.extract_media = BasePlatformAdapter.extract_media
        adapter.extract_images = BasePlatformAdapter.extract_images
        adapter.extract_local_files = BasePlatformAdapter.extract_local_files
        runner.adapters[Platform.SLACK] = adapter
        runner._run_in_executor_with_context = AsyncMock(
            return_value={"final_response": "private result", "messages": []}
        )
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "test-key"})
        )
        runner._resolve_session_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": "test-model",
                "runtime": {"api_key": "test-key"},
                "request_overrides": None,
            }
        )
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")
        job_state = {"source": source, "private_user_id": "U1"}

        with patch("gateway.run._load_gateway_config", return_value={}):
            await runner._run_background_task(
                "work",
                source,
                "bg_private",
                job_state=job_state,
            )

        adapter.send_private_notice.assert_awaited_once()
        adapter._send_with_retry.assert_not_awaited()
        assert runner._background_task_outcomes["bg_private"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_private_background_media_is_not_posted_to_channel(self):
        runner = _make_runner()
        adapter = MagicMock()
        adapter.send_private_notice = AsyncMock(return_value=SendResult(success=True))
        adapter._send_with_retry = AsyncMock()
        adapter.extract_media = MagicMock(return_value=(["/tmp/private.pdf"], ""))
        adapter.extract_images = MagicMock(return_value=([], ""))
        adapter.extract_local_files = MagicMock(return_value=([], ""))
        runner.adapters[Platform.SLACK] = adapter
        runner._run_in_executor_with_context = AsyncMock(
            return_value={"final_response": "MEDIA:/tmp/private.pdf", "messages": []}
        )
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "test-key"})
        )
        runner._resolve_session_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": "test-model",
                "runtime": {"api_key": "test-key"},
                "request_overrides": None,
            }
        )
        runner._deliver_media_from_response = AsyncMock()
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")

        with patch("gateway.run._load_gateway_config", return_value={}):
            await runner._run_background_task(
                "work",
                source,
                "bg_private_media",
                job_state={"source": source, "private_user_id": "U1"},
            )

        runner._deliver_media_from_response.assert_not_awaited()
        sent = [call.kwargs["content"] for call in adapter.send_private_notice.await_args_list]
        assert any("did not post it to the channel" in content for content in sent)
        assert all("/tmp/private.pdf" not in content for content in sent)

    def test_background_outcomes_are_bounded_and_persisted(self, tmp_path):
        runner = _make_runner()
        runner._background_outcomes_path = tmp_path / "outcomes.json"
        runner._background_task_outcomes = {}

        runner._background_task_outcomes = {
            f"bg_{index}": {"status": "success"} for index in range(512)
        }
        runner._record_background_task_outcome("bg_512", "success")

        assert len(runner._background_task_outcomes) == 512
        assert "bg_0" not in runner._background_task_outcomes
        persisted = json.loads(runner._background_outcomes_path.read_text())
        assert persisted == runner._background_task_outcomes
        assert runner._load_background_task_outcomes() == persisted

    def test_later_media_warning_does_not_replace_failed_text_result(self):
        runner = _make_runner()
        runner._background_task_outcomes = {}
        answer = {"content": "the actual answer"}
        warning = {"content": "attachment was blocked"}

        runner._record_background_task_outcome(
            "bg_mixed",
            "delivery_failed",
            detail="text_result",
            pending_notice=answer,
        )
        runner._record_background_task_outcome(
            "bg_mixed",
            "delivery_failed",
            detail="private_media_blocked",
            pending_notice=warning,
        )
        runner._record_background_task_outcome(
            "bg_mixed",
            "delivery_failed_notified",
            detail="private_media_blocked",
        )

        assert runner._background_task_outcomes["bg_mixed"]["pending_notice"] == answer

    @pytest.mark.asyncio
    async def test_background_notice_adds_secondary_workspace_to_live_and_retry_route(self):
        runner = _make_runner()
        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=False))
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")
        job = {"source": source, "team_id": "T2"}

        await runner._send_background_job_notice(
            adapter,
            job,
            "shutdown notice",
            {"thread_id": "root-1"},
        )
        pending = runner._background_pending_notice(
            job,
            "shutdown notice",
            {"thread_id": "root-1"},
        )

        assert adapter._send_with_retry.await_args.kwargs["metadata"] == {
            "thread_id": "root-1",
            "team_id": "T2",
        }
        assert adapter._send_with_retry.await_args.kwargs["persist_failure"] is False
        assert pending["metadata"] == {"thread_id": "root-1", "team_id": "T2"}

    @pytest.mark.asyncio
    async def test_failed_background_result_persists_route_and_retries(self, tmp_path):
        runner = _make_runner()
        runner._background_outcomes_path = tmp_path / "outcomes.json"
        runner._background_task_outcomes = {}
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SendResult(success=True))
        runner.adapters[Platform.SLACK] = adapter
        notice = {
            "platform": "slack",
            "chat_id": "C1",
            "private_user_id": None,
            "content": "Background task result: done",
            "metadata": {"thread_id": "thread-1", "team_id": "T2"},
        }

        runner._record_background_task_outcome(
            "bg_retry",
            "delivery_failed",
            detail="text_result",
            pending_notice=notice,
        )

        persisted = json.loads(runner._background_outcomes_path.read_text())
        assert persisted["bg_retry"]["pending_notice"] == notice
        restarted = _make_runner()
        restarted._background_outcomes_path = runner._background_outcomes_path
        restarted._background_task_outcomes = restarted._load_background_task_outcomes()
        restarted.adapters[Platform.SLACK] = adapter
        assert await restarted._retry_background_task_outcomes() == 1
        adapter.send.assert_awaited_once_with(
            "C1",
            "Background task result: done",
            metadata={"thread_id": "thread-1", "team_id": "T2"},
        )
        assert restarted._background_task_outcomes["bg_retry"]["status"] == "retry_delivered"
        assert "pending_notice" not in restarted._background_task_outcomes["bg_retry"]

    @pytest.mark.asyncio
    async def test_text_completion_uses_retrying_delivery(self):
        """A retryable completion failure reaches the user on retry."""
        runner = _make_runner()

        class RetryAdapter(BasePlatformAdapter):
            def __init__(self):
                super().__init__(MagicMock(), Platform.TELEGRAM)
                self.results = [
                    SendResult(success=False, error="connection reset", retryable=True),
                    SendResult(success=True, message_id="completion-2"),
                ]
                self.calls = []

            async def connect(self):
                return True

            async def disconnect(self):
                return None

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                self.calls.append((chat_id, content, metadata))
                return self.results.pop(0)

            async def get_chat_info(self, chat_id):
                return {"name": chat_id, "type": "dm"}

        adapter = RetryAdapter()
        runner.adapters[Platform.TELEGRAM] = adapter
        runner._run_in_executor_with_context = AsyncMock(
            return_value={"final_response": "done", "messages": []}
        )
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "test-key"})
        )
        runner._resolve_session_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": "test-model",
                "runtime": {"api_key": "test-key"},
                "request_overrides": None,
            }
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
        )

        with patch("gateway.run._load_gateway_config", return_value={}), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await runner._run_background_task("say hello", source, "bg_test")

        assert len(adapter.calls) == 2
        assert "Background task result" in adapter.calls[-1][1]

    @pytest.mark.asyncio
    async def test_failed_completion_emits_terminal_delivery_evidence(self, caplog):
        runner = _make_runner()
        adapter = MagicMock()
        adapter.extract_media = BasePlatformAdapter.extract_media
        adapter.extract_images = BasePlatformAdapter.extract_images
        adapter.extract_local_files = BasePlatformAdapter.extract_local_files
        adapter._send_with_retry = AsyncMock(
            return_value=SendResult(success=False, error="delivery exhausted")
        )
        runner.adapters[Platform.SLACK] = adapter
        runner._run_in_executor_with_context = AsyncMock(
            return_value={"final_response": "done", "messages": []}
        )
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "test-key"})
        )
        runner._resolve_session_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": "test-model",
                "runtime": {"api_key": "test-key"},
                "request_overrides": None,
            }
        )
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")

        with patch("gateway.run._load_gateway_config", return_value={}):
            await runner._run_background_task("work", source, "bg_test")

        assert "background_task task_id=bg_test" in caplog.text
        assert "phase=text_result delivery_success=false" in caplog.text
        assert "error=delivery exhausted" in caplog.text
        assert runner._background_task_outcomes["bg_test"]["status"] == "delivery_failed"

    @pytest.mark.asyncio
    async def test_agent_error_is_not_labeled_successful(self):
        runner = _make_runner()
        adapter = MagicMock()
        adapter.extract_media = BasePlatformAdapter.extract_media
        adapter.extract_images = BasePlatformAdapter.extract_images
        adapter.extract_local_files = BasePlatformAdapter.extract_local_files
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
        runner.adapters[Platform.SLACK] = adapter
        runner._run_in_executor_with_context = AsyncMock(
            return_value={"final_response": "", "error": "provider failed"}
        )
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "test-key"})
        )
        runner._resolve_session_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": "test-model",
                "runtime": {"api_key": "test-key"},
                "request_overrides": None,
            }
        )
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")

        with patch("gateway.run._load_gateway_config", return_value={}):
            await runner._run_background_task("work", source, "bg_test")

        content = adapter._send_with_retry.await_args.kwargs["content"]
        assert content.startswith("❌ Background task failed")
        assert "✅" not in content
        assert runner._background_task_outcomes["bg_test"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_empty_result_is_not_labeled_successful(self):
        runner = _make_runner()
        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
        runner.adapters[Platform.SLACK] = adapter
        runner._run_in_executor_with_context = AsyncMock(return_value={})
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "test-key"})
        )
        runner._resolve_session_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": "test-model",
                "runtime": {"api_key": "test-key"},
                "request_overrides": None,
            }
        )
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")

        with patch("gateway.run._load_gateway_config", return_value={}):
            await runner._run_background_task("work", source, "bg_test")

        content = adapter._send_with_retry.await_args.kwargs["content"]
        assert content.startswith("⚠️ Background task finished without a response")
        assert "✅" not in content
        assert runner._background_task_outcomes["bg_test"]["status"] == "empty"

    @pytest.mark.asyncio
    async def test_shutdown_cancels_then_notifies_user_background_job(self):
        runner = _make_runner()
        runner._user_background_jobs = {}
        runner._background_task_outcomes = {}
        adapter = MagicMock()
        runner.adapters[Platform.SLACK] = adapter
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")

        started = asyncio.Event()

        async def pending_job():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(pending_job())
        await started.wait()
        runner._user_background_jobs[task] = {
            "task_id": "bg_shutdown",
            "source": source,
            "event_message_id": "thread-1",
        }

        async def assert_cancelled_before_notice(**_kwargs):
            assert task.cancelled()
            return SendResult(success=True)

        adapter._send_with_retry = AsyncMock(side_effect=assert_cancelled_before_notice)

        await runner._cancel_user_background_jobs_for_shutdown()

        assert task.cancelled()
        adapter._send_with_retry.assert_awaited_once()
        assert "shutting down" in adapter._send_with_retry.await_args.kwargs["content"]
        assert runner._background_task_outcomes["bg_shutdown"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_shutdown_interrupts_agent_before_cancelling_worker(self):
        runner = _make_runner()
        runner._user_background_jobs = {}
        runner._background_task_outcomes = {}
        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
        runner.adapters[Platform.SLACK] = adapter
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")
        started = asyncio.Event()
        thread_done = threading.Event()
        agent = MagicMock()

        async def pending_job():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                thread_done.set()

        task = asyncio.create_task(pending_job())
        await started.wait()
        runner._user_background_jobs[task] = {
            "task_id": "bg_interrupt",
            "source": source,
            "event_message_id": None,
            "agent": agent,
            "thread_started": threading.Event(),
            "thread_done": thread_done,
        }
        runner._user_background_jobs[task]["thread_started"].set()

        await runner._cancel_user_background_jobs_for_shutdown()

        agent.interrupt.assert_called_once_with("Gateway shutting down")
        assert thread_done.is_set()
        assert runner._background_task_outcomes["bg_interrupt"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_shutdown_reports_unconfirmed_worker_termination(self):
        runner = _make_runner()
        runner._BACKGROUND_SHUTDOWN_WAIT_SECONDS = 0
        runner._user_background_jobs = {}
        runner._background_task_outcomes = {}
        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
        runner.adapters[Platform.SLACK] = adapter
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")
        started = asyncio.Event()
        thread_started = threading.Event()
        thread_started.set()

        async def pending_job():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(pending_job())
        await started.wait()
        runner._user_background_jobs[task] = {
            "task_id": "bg_unconfirmed",
            "source": source,
            "event_message_id": None,
            "agent": MagicMock(),
            "thread_started": thread_started,
            "thread_done": threading.Event(),
        }

        with patch("asyncio.to_thread", new=AsyncMock()) as wait_call:
            await runner._cancel_user_background_jobs_for_shutdown()

        wait_call.assert_not_awaited()
        content = adapter._send_with_retry.await_args.kwargs["content"]
        assert "termination is not confirmed" in content
        assert (
            runner._background_task_outcomes["bg_unconfirmed"]["status"]
            == "interrupt_unconfirmed_notified"
        )

    @pytest.mark.asyncio
    async def test_shutdown_does_not_cancel_job_that_suppresses_cancellation(self):
        runner = _make_runner()
        runner._user_background_jobs = {}
        runner._background_task_outcomes = {}
        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
        runner.adapters[Platform.SLACK] = adapter
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")
        started = asyncio.Event()

        async def finish_on_cancel():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                runner._record_background_task_outcome("bg_race", "success")

        task = asyncio.create_task(finish_on_cancel())
        await started.wait()
        runner._user_background_jobs[task] = {
            "task_id": "bg_race",
            "source": source,
            "event_message_id": None,
        }

        await runner._cancel_user_background_jobs_for_shutdown()

        assert task.done()
        assert not task.cancelled()
        adapter._send_with_retry.assert_not_awaited()
        assert runner._background_task_outcomes["bg_race"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_shutdown_retains_undelivered_cancellation_outcome(self):
        runner = _make_runner()
        runner._user_background_jobs = {}
        runner._background_task_outcomes = {}
        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock(
            return_value=SendResult(success=False, error="offline")
        )
        runner.adapters[Platform.SLACK] = adapter
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")

        started = asyncio.Event()

        async def pending_job():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(pending_job())
        await started.wait()
        runner._user_background_jobs[task] = {
            "task_id": "bg_shutdown_failed",
            "source": source,
            "event_message_id": None,
        }

        await runner._cancel_user_background_jobs_for_shutdown()

        assert task.cancelled()
        assert (
            runner._background_task_outcomes["bg_shutdown_failed"]["status"]
            == "cancelled_undelivered"
        )

    @pytest.mark.asyncio
    async def test_running_task_preserves_recorded_shutdown_outcome_on_cancel(self):
        runner = _make_runner()
        runner._background_task_outcomes = {}
        adapter = MagicMock()
        runner.adapters[Platform.SLACK] = adapter
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "test-key"})
        )
        runner._resolve_session_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": "test-model",
                "runtime": {"api_key": "test-key"},
                "request_overrides": None,
            }
        )
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")
        started = asyncio.Event()

        async def wait_for_cancel(_callable):
            started.set()
            await asyncio.Event().wait()

        runner._run_in_executor_with_context = AsyncMock(side_effect=wait_for_cancel)

        with patch("gateway.run._load_gateway_config", return_value={}):
            task = asyncio.create_task(
                runner._run_background_task("work", source, "bg_cancelled")
            )
            await started.wait()
            runner._record_background_task_outcome(
                "bg_cancelled",
                "cancelled_undelivered",
                detail="gateway shutdown",
            )
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert (
            runner._background_task_outcomes["bg_cancelled"]["status"]
            == "cancelled_undelivered"
        )

    @pytest.mark.asyncio
    async def test_bare_local_artifact_is_uploaded_without_exposing_path(self, tmp_path):
        runner = _make_runner()
        artifact = tmp_path / "report.pdf"
        artifact.write_bytes(b"report")
        adapter = MagicMock()
        adapter.name = "Slack"
        adapter.extract_media = BasePlatformAdapter.extract_media
        adapter.extract_images = BasePlatformAdapter.extract_images
        adapter.extract_local_files = BasePlatformAdapter.extract_local_files
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
        runner.adapters[Platform.SLACK] = adapter
        runner._run_in_executor_with_context = AsyncMock(
            return_value={"final_response": f"Finished: {artifact}", "messages": []}
        )
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "test-key"})
        )
        runner._resolve_session_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": "test-model",
                "runtime": {"api_key": "test-key"},
                "request_overrides": None,
            }
        )
        runner._deliver_media_from_response = AsyncMock(return_value=True)
        source = SessionSource(
            platform=Platform.SLACK,
            user_id="U1",
            chat_id="C1",
        )

        with patch("gateway.run._load_gateway_config", return_value={}):
            await runner._run_background_task("make report", source, "bg_test")

        runner._deliver_media_from_response.assert_awaited_once()
        sent_text = adapter._send_with_retry.await_args.kwargs["content"]
        assert str(artifact) not in sent_text
        assert "Finished:" in sent_text

    @pytest.mark.asyncio
    async def test_media_only_upload_failure_sends_visible_notice(self):
        runner = _make_runner()
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "test-key"})
        )
        runner._resolve_session_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": "test-model",
                "runtime": {"api_key": "test-key"},
                "request_overrides": None,
            }
        )
        runner._run_in_executor_with_context = AsyncMock(
            return_value={
                "final_response": "MEDIA:/tmp/report.pdf",
                "messages": [],
            }
        )

        adapter = MagicMock()
        adapter.name = "Slack"
        adapter.extract_media = BasePlatformAdapter.extract_media
        adapter.extract_images = BasePlatformAdapter.extract_images
        adapter.extract_local_files = BasePlatformAdapter.extract_local_files
        adapter.send_document = AsyncMock(
            return_value=SendResult(success=False, error="upload failed")
        )
        adapter.send_video = AsyncMock(return_value=SendResult(success=True))
        adapter.send_voice = AsyncMock(return_value=SendResult(success=True))
        adapter.send_multiple_images = AsyncMock(return_value=SendResult(success=True))
        adapter._send_with_retry = AsyncMock(
            return_value=SendResult(success=True, message_id="notice-1")
        )
        adapter.send = AsyncMock(return_value=SendResult(success=True))
        runner.adapters[Platform.SLACK] = adapter

        source = SessionSource(
            platform=Platform.SLACK,
            user_id="U123",
            chat_id="C123",
            user_name="testuser",
        )

        with patch("gateway.run._load_gateway_config", return_value={}):
            await runner._run_background_task(
                "make a report",
                source,
                "bg_test",
            )

        adapter.send_document.assert_awaited_once()
        adapter._send_with_retry.assert_awaited_once()
        notice = adapter._send_with_retry.await_args.kwargs["content"]
        assert "report.pdf" in notice
        assert "was not attached" in notice

    @pytest.mark.asyncio
    async def test_telegram_dm_topic_completion_preserves_reply_anchor_metadata(self, monkeypatch):
        """Background completion metadata must let Telegram send thread id plus reply id."""
        from gateway import run as gateway_run

        runner = _make_runner()
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "test-key"})
        )
        runner._resolve_session_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": "test-model",
                "runtime": {"api_key": "test-key"},
                "request_overrides": None,
            }
        )
        runner._run_in_executor_with_context = AsyncMock(
            return_value={"final_response": "done", "messages": []}
        )
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})

        mock_adapter = AsyncMock()
        mock_adapter._send_with_retry = AsyncMock(
            return_value=SendResult(success=True, message_id="completion-1")
        )
        mock_adapter.extract_media = MagicMock(return_value=([], "done"))
        mock_adapter.extract_images = MagicMock(return_value=([], "done"))
        mock_adapter.extract_local_files = MagicMock(return_value=([], "done"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            chat_type="dm",
            thread_id="20197",
        )

        await runner._run_background_task(
            "say hello",
            source,
            "bg_test",
            event_message_id="463",
        )

        mock_adapter._send_with_retry.assert_awaited_once()
        assert mock_adapter._send_with_retry.call_args.kwargs["metadata"] == {
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "direct_messages_topic_id": "20197",
            "telegram_reply_to_message_id": "463",
        }

    @pytest.mark.asyncio
    async def test_agent_cleanup_runs_when_background_agent_raises(self):
        """Temporary background agents must be cleaned up on error paths too."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter._send_with_retry = AsyncMock(
            return_value=SendResult(success=True, message_id="failure-1")
        )
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.shutdown_memory_provider = MagicMock()
            mock_agent_instance.close = MagicMock()
            mock_agent_instance.run_conversation.side_effect = RuntimeError("boom")
            MockAgent.return_value = mock_agent_instance

            await runner._run_background_task("say hello", source, "bg_test")

        mock_adapter._send_with_retry.assert_awaited_once()
        mock_agent_instance.shutdown_memory_provider.assert_called_once()
        mock_agent_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_sends_error_message(self):
        """When the agent raises an exception, an error message is sent."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter._send_with_retry = AsyncMock(
            return_value=SendResult(success=True, message_id="failure-1")
        )
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", side_effect=RuntimeError("boom")):
            await runner._run_background_task("test prompt", source, "bg_test")

        mock_adapter._send_with_retry.assert_awaited_once()
        call_args = mock_adapter._send_with_retry.call_args
        content = call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "failed" in content.lower()

    @pytest.mark.asyncio
    async def test_exception_preserves_delivered_failure_notice(self):
        runner = _make_runner()
        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock(
            return_value=SendResult(
                success=False,
                error="delivery exhausted",
                failure_notice_delivered=True,
            )
        )
        runner.adapters[Platform.SLACK] = adapter
        source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1")

        with patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            side_effect=RuntimeError("boom"),
        ):
            await runner._run_background_task("work", source, "bg_exception")

        assert (
            runner._background_task_outcomes["bg_exception"]["status"]
            == "delivery_failed_notified"
        )


# ---------------------------------------------------------------------------
# /background in help and known_commands
# ---------------------------------------------------------------------------


class TestBackgroundInHelp:
    """Verify /background appears in help text and known commands."""

    @pytest.mark.asyncio
    async def test_background_in_help_output(self):
        """The /help output includes /background."""
        runner = _make_runner()
        event = _make_event(text="/help")
        result = await runner._handle_help_command(event)
        assert "/background" in result

    def test_background_is_known_command(self):
        """The /background command is in GATEWAY_KNOWN_COMMANDS."""
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "background" in GATEWAY_KNOWN_COMMANDS

    def test_bg_alias_is_known_command(self):
        """The /bg alias is in GATEWAY_KNOWN_COMMANDS."""
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "bg" in GATEWAY_KNOWN_COMMANDS


# ---------------------------------------------------------------------------
# CLI /background command definition
# ---------------------------------------------------------------------------


class TestBackgroundInCLICommands:
    """Verify /background is registered in the CLI command system."""

    def test_background_in_commands_dict(self):
        """The /background command is in the COMMANDS dict."""
        from hermes_cli.commands import COMMANDS
        assert "/background" in COMMANDS

    def test_bg_alias_in_commands_dict(self):
        """The /bg alias is in the COMMANDS dict."""
        from hermes_cli.commands import COMMANDS
        assert "/bg" in COMMANDS

    def test_background_in_session_category(self):
        """The /background command is in the Session category."""
        from hermes_cli.commands import COMMANDS_BY_CATEGORY
        assert "/background" in COMMANDS_BY_CATEGORY["Session"]

    def test_background_autocompletes(self):
        """The /background command appears in autocomplete results."""
        pytest.importorskip("prompt_toolkit")
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document

        completer = SlashCommandCompleter()
        doc = Document("backgro")  # Partial match
        completions = list(completer.get_completions(doc, None))
        # Text doesn't start with / so no completions
        assert len(completions) == 0

        doc = Document("/backgro")  # With slash prefix
        completions = list(completer.get_completions(doc, None))
        cmd_displays = [str(c.display) for c in completions]
        assert any("/background" in d for d in cmd_displays)
