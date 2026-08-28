"""Tests for topic-aware gateway progress updates."""

import asyncio
import importlib
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.session import SessionEntry, SessionSource, build_session_key


class ProgressCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.typing = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": {"stopped": True}})

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class SmallLimitProgressAdapter(ProgressCaptureAdapter):
    """Adapter with a tiny platform limit to exercise progress rollover."""

    MAX_MESSAGE_LENGTH = 180

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self._next_id = 0
        self.oversized_edits = []
        self.oversized_sends = []

    def _mint_id(self):
        self._next_id += 1
        return f"progress-{self._next_id}"

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        if len(content) > self.MAX_MESSAGE_LENGTH:
            self.oversized_sends.append(content)
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=self._mint_id())

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        if len(content) > self.MAX_MESSAGE_LENGTH:
            self.oversized_edits.append(content)
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        return SendResult(success=True, message_id=message_id)


class MetadataEditProgressCaptureAdapter(ProgressCaptureAdapter):
    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=message_id)


class NonEditingProgressCaptureAdapter(ProgressCaptureAdapter):
    SUPPORTS_MESSAGE_EDITING = False

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        raise AssertionError("non-editable adapters should not receive edit_message calls")


class FakeAgent:
    def __init__(self, **kwargs):
        # Capture anything passed via kwargs (older code path) but don't
        # freeze it — production now assigns tool_progress_callback after
        # construction (see gateway/run.py around the agent-cache hit),
        # so we must read it at call time, not at init.
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "terminal", "pwd", {})
            time.sleep(0.35)
            cb("tool.started", "browser_navigate", "https://example.com", {})
            time.sleep(0.35)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class LongPreviewAgent:
    """Agent that emits a tool call with a very long preview string."""
    LONG_CMD = "cd /home/teknium/.hermes/hermes-agent/.worktrees/hermes-d8860339 && source .venv/bin/activate && python -m pytest tests/gateway/test_run_progress_topics.py -n0 -q"

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback("tool.started", "terminal", self.LONG_CMD, {})
        time.sleep(0.35)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class DelayedProgressAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback("tool.started", "terminal", "first command", {})
        time.sleep(0.45)
        self.tool_progress_callback("tool.started", "terminal", "second command", {})
        time.sleep(0.1)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class ManyProgressLinesAgent:
    """Emits enough tool-progress lines to exceed a single platform bubble."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        assert cb is not None
        cb("tool.started", "terminal", "first-short", {})
        # Let the progress task create the first editable bubble, then enqueue
        # the rest quickly.  The cancellation drain must roll them into fresh
        # editable bubbles instead of trying to edit the first one past limit.
        time.sleep(0.35)
        for idx in range(1, 8):
            cb("tool.started", "terminal", f"overflow-line-{idx}-" + "x" * 45, {})
        time.sleep(0.1)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class DelayedInterimAgent:
    def __init__(self, **kwargs):
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.interim_assistant_callback("first interim")
        time.sleep(0.45)
        self.interim_assistant_callback("second interim")
        time.sleep(0.1)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


@pytest.mark.asyncio
async def test_run_agent_progress_stays_in_originating_topic(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 - register terminal emoji for this fake-agent test

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key="agent:main:telegram:group:-1001:17585",
    )

    assert result["final_response"] == "done"
    assert adapter.sent == [
        {
            "chat_id": "-1001",
            "content": '💻 terminal: "pwd"',
            "reply_to": None,
            "metadata": {"thread_id": "17585"},
        }
    ]
    assert adapter.edits
    assert all(call["metadata"] == {"thread_id": "17585"} for call in adapter.typing)


@pytest.mark.asyncio
async def test_run_agent_progress_edits_keep_originating_topic_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = MetadataEditProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-progress-edit-topic",
        session_key="agent:main:telegram:group:-1001:17585",
    )

    assert result["final_response"] == "done"
    assert adapter.edits
    assert all(call["metadata"] == {"thread_id": "17585"} for call in adapter.edits)


@pytest.mark.asyncio
async def test_run_agent_progress_does_not_use_event_message_id_for_telegram_dm(monkeypatch, tmp_path):
    """Telegram DM progress must not reuse event message id as thread metadata."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-2",
        session_key="agent:main:telegram:dm:12345",
        event_message_id="777",
    )

    assert result["final_response"] == "done"
    assert adapter.sent
    assert adapter.sent[0]["metadata"] is None
    assert all(call["metadata"] is None for call in adapter.typing)


@pytest.mark.asyncio
async def test_run_agent_progress_uses_event_message_id_for_slack_dm(monkeypatch, tmp_path):
    """Slack DM progress should keep event ts fallback threading."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    # Since PR #8006, Slack's built-in display tier sets tool_progress="off"
    # by default. Override via config so this test still exercises the
    # progress-callback path the Slack DM event_message_id threading depends on.
    import yaml
    (tmp_path / "config.yaml").write_text(
        yaml.dump({"display": {"platforms": {"slack": {"tool_progress": "all"}}}}),
        encoding="utf-8",
    )

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="D123",
        chat_type="dm",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-3",
        session_key="agent:main:slack:dm:D123",
        event_message_id="1234567890.000001",
    )

    assert result["final_response"] == "done"
    assert adapter.sent
    assert adapter.sent[0]["metadata"] == {"thread_id": "1234567890.000001"}
    assert all(call["metadata"] == {"thread_id": "1234567890.000001"} for call in adapter.typing)


@pytest.mark.asyncio
async def test_run_agent_feishu_progress_replies_inside_existing_thread(monkeypatch, tmp_path):
    """Feishu needs reply_to plus reply_in_thread metadata for topic-scoped progress."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        thread_id="topic_17585",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-feishu-progress",
        session_key="agent:main:feishu:group:oc_chat:topic_17585",
        event_message_id="om_triggering_user_message",
    )

    assert result["final_response"] == "done"
    assert adapter.sent
    assert adapter.sent[0]["reply_to"] == "om_triggering_user_message"
    assert adapter.sent[0]["metadata"] == {"thread_id": "topic_17585"}
    assert adapter.edits
    assert adapter.edits[0]["message_id"] == "progress-1"


# ---------------------------------------------------------------------------
# Preview truncation tests (all/new mode respects tool_preview_length)
# ---------------------------------------------------------------------------


def _run_long_preview_helper(monkeypatch, tmp_path, preview_length=0):
    """Shared setup for long-preview truncation tests.

    Returns (adapter, result) after running the agent with LongPreviewAgent.
    ``preview_length`` controls display.tool_preview_length in the config file
    that _run_agent reads — so the gateway picks it up the same way production does.
    """
    import asyncio
    import yaml

    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = LongPreviewAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    # Write config.yaml so _run_agent picks up tool_preview_length
    config = {"display": {"tool_preview_length": preview_length}}
    (tmp_path / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        thread_id=None,
    )

    result = asyncio.get_event_loop().run_until_complete(
        runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-trunc",
            session_key="agent:main:telegram:dm:12345",
        )
    )
    return adapter, result


def test_all_mode_default_truncation_40_chars(monkeypatch, tmp_path):
    """When tool_preview_length is 0 (default), all/new mode truncates to 40 chars."""
    adapter, result = _run_long_preview_helper(monkeypatch, tmp_path, preview_length=0)
    assert result["final_response"] == "done"
    assert adapter.sent
    content = adapter.sent[0]["content"]
    # The long command should be truncated — total preview <= 40 chars
    assert "..." in content
    # Extract the preview part between quotes
    import re
    match = re.search(r'"(.+)"', content)
    assert match, f"No quoted preview found in: {content}"
    preview_text = match.group(1)
    assert len(preview_text) <= 40, f"Preview too long ({len(preview_text)}): {preview_text}"


def test_all_mode_respects_custom_preview_length(monkeypatch, tmp_path):
    """When tool_preview_length is explicitly set (e.g. 120), all/new mode uses that."""
    adapter, result = _run_long_preview_helper(monkeypatch, tmp_path, preview_length=120)
    assert result["final_response"] == "done"
    assert adapter.sent
    content = adapter.sent[0]["content"]
    # With 120-char cap, the command (165 chars) should still be truncated but longer
    import re
    match = re.search(r'"(.+)"', content)
    assert match, f"No quoted preview found in: {content}"
    preview_text = match.group(1)
    # Should be longer than the 40-char default
    assert len(preview_text) > 40, f"Preview suspiciously short ({len(preview_text)}): {preview_text}"
    # But still capped at 120
    assert len(preview_text) <= 120, f"Preview too long ({len(preview_text)}): {preview_text}"


def test_all_mode_no_truncation_when_preview_fits(monkeypatch, tmp_path):
    """Short previews (under the cap) are not truncated."""
    # Set a generous cap — the LongPreviewAgent's command is ~165 chars
    adapter, result = _run_long_preview_helper(monkeypatch, tmp_path, preview_length=200)
    assert result["final_response"] == "done"
    assert adapter.sent
    content = adapter.sent[0]["content"]
    # With a 200-char cap, the 165-char command should NOT be truncated
    assert "..." not in content, f"Preview was truncated when it shouldn't be: {content}"


class CommentaryAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.interim_assistant_callback:
            self.interim_assistant_callback("I'll inspect the repo first.", already_streamed=False)
        time.sleep(0.1)
        if self.stream_delta_callback:
            self.stream_delta_callback("done")
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class PreviewedResponseAgent:
    def __init__(self, **kwargs):
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.interim_assistant_callback:
            self.interim_assistant_callback("You're welcome.", already_streamed=False)
        return {
            "final_response": "You're welcome.",
            "response_previewed": True,
            "messages": [],
            "api_calls": 1,
        }


class StreamingRefineAgent:
    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.stream_delta_callback:
            self.stream_delta_callback("Continuing to refine:")
        time.sleep(0.1)
        if self.stream_delta_callback:
            self.stream_delta_callback(" Final answer.")
        return {
            "final_response": "Continuing to refine: Final answer.",
            "response_previewed": True,
            "messages": [],
            "api_calls": 1,
        }


class QueuedCommentaryAgent:
    calls = 0

    def __init__(self, **kwargs):
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        if type(self).calls == 1 and self.interim_assistant_callback:
            self.interim_assistant_callback("I'll inspect the repo first.", already_streamed=False)
        return {
            "final_response": f"final response {type(self).calls}",
            "messages": [],
            "api_calls": 1,
        }


class EveryTurnCommentaryAgent(QueuedCommentaryAgent):
    calls = 0

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        if self.interim_assistant_callback:
            self.interim_assistant_callback(
                f"private commentary {type(self).calls}",
                already_streamed=False,
            )
        return {
            "final_response": f"final response {type(self).calls}",
            "messages": [],
            "api_calls": 1,
        }


class EarlyReturnSteerAgent:
    """Simulate a runtime branch that returns before the loop-level drain."""

    calls = 0
    messages = []

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        type(self).messages.append(message)
        return {
            "final_response": f"final response {type(self).calls}",
            "messages": [],
            "api_calls": 1,
        }

    def _close_steering(self):
        if type(self).calls == 1:
            return "late steer"
        return None


class EmptyEarlyReturnSteerAgent(EarlyReturnSteerAgent):
    """Return no text before a late steer becomes a second turn."""

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        type(self).messages.append(message)
        return {
            "final_response": "" if type(self).calls == 1 else "steer response",
            "messages": [],
            "api_calls": 1,
        }


class MediaOnlyAgent:
    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        return {
            "final_response": "",
            "messages": [
                {
                    "role": "tool",
                    "content": '{"artifact": "MEDIA:/tmp/report.pdf"}',
                }
            ],
            "api_calls": 1,
        }


class QueuedMediaAgent:
    calls = 0

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        if type(self).calls == 1:
            return {
                "final_response": "",
                "messages": [
                    {
                        "role": "tool",
                        "content": '{"artifact": "MEDIA:/tmp/queued-report.pdf"}',
                    }
                ],
                "api_calls": 1,
            }
        return {
            "final_response": "queued response",
            "messages": [],
            "api_calls": 1,
        }


class QueuedMarkdownImageAgent:
    calls = 0

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        return {
            "final_response": (
                "![chart](https://example.com/chart.png)"
                if type(self).calls == 1
                else "queued response"
            ),
            "messages": [],
            "api_calls": 1,
        }


class QueuedFailedMediaAgent:
    calls = 0

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        return {
            "final_response": (
                "Report attached.\nMEDIA:/tmp/report.pdf"
                if type(self).calls == 1
                else "queued response"
            ),
            "messages": [],
            "api_calls": 1,
        }


class QueuedFailedImageBatchAgent:
    calls = 0

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        return {
            "final_response": (
                "Charts attached.\n"
                "![first](https://example.com/first.png)\n"
                "![second](https://example.com/second.png)"
                if type(self).calls == 1
                else "queued response"
            ),
            "messages": [],
            "api_calls": 1,
        }


class SecondCallRaisingAgent:
    calls = 0

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        if type(self).calls == 2:
            raise RuntimeError("queued agent failed")
        return {
            "final_response": "first response",
            "messages": [],
            "api_calls": 1,
        }


class ImageCaptureAdapter(ProgressCaptureAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.image_batches = []

    async def send_multiple_images(
        self,
        chat_id,
        images,
        reply_to=None,
        metadata=None,
        human_delay=0.0,
    ):
        self.image_batches.append(images)
        return SendResult(success=True, message_id="image-1")


class FailedMediaNoticeAdapter(ProgressCaptureAdapter):
    async def send_document(self, chat_id, file_path, reply_to=None, metadata=None):
        return SendResult(success=False, error="upload failed")

    async def _send_with_retry(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        if content.startswith("⚠️"):
            return SendResult(success=False, error="notice failed")
        return SendResult(success=True, message_id="text-1")


class FailedImageBatchAdapter(ProgressCaptureAdapter):
    async def send_multiple_images(
        self,
        chat_id,
        images,
        reply_to=None,
        metadata=None,
        human_delay=0.0,
    ):
        return SendResult(success=False, error="one image failed")


class RaisingAgent:
    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        raise RuntimeError("agent failed")


class BackgroundReviewAgent:
    def __init__(self, **kwargs):
        self.background_review_callback = kwargs.get("background_review_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.background_review_callback:
            self.background_review_callback("💾 Skill 'prospect-scanner' created.")
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class VerboseAgent:
    """Agent that emits a tool call with args whose JSON exceeds 200 chars."""
    LONG_CODE = "x" * 300

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback(
            "tool.started", "execute_code", None,
            {"code": self.LONG_CODE},
        )
        time.sleep(0.35)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


async def _run_with_agent(
    monkeypatch,
    tmp_path,
    agent_cls,
    *,
    session_id,
    pending_text=None,
    config_data=None,
    platform=Platform.TELEGRAM,
    chat_id="-1001",
    chat_type="group",
    thread_id="17585",
    adapter_cls=ProgressCaptureAdapter,
    reply_event=None,
    pending_event=None,
    overflow_events=None,
    steer_events=None,
    draining=False,
    interrupt_depth=0,
):
    if config_data:
        import yaml

        (tmp_path / "config.yaml").write_text(yaml.dump(config_data), encoding="utf-8")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = adapter_cls(platform=platform)
    runner = _make_runner(adapter)
    adapter._test_runner = runner
    runner._draining = draining
    gateway_run = importlib.import_module("gateway.run")
    if config_data and "streaming" in config_data:
        runner.config.streaming = StreamingConfig.from_dict(config_data["streaming"])
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    source = SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        thread_id=thread_id,
    )
    session_key = f"agent:main:{platform.value}:{chat_type}:{chat_id}"
    if thread_id:
        session_key = f"{session_key}:{thread_id}"
    if pending_event is not None:
        adapter._pending_messages[session_key] = pending_event
    elif pending_text is not None:
        adapter._pending_messages[session_key] = MessageEvent(
            text=pending_text,
            message_type=MessageType.TEXT,
            source=source,
            message_id="queued-1",
        )
    if overflow_events:
        runner._queued_events = {session_key: list(overflow_events)}
    if steer_events:
        runner._steer_reply_events = {session_key: list(steer_events)}

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id=session_id,
        session_key=session_key,
        reply_event=reply_event,
        _interrupt_depth=interrupt_depth,
    )
    return adapter, result


@pytest.mark.asyncio
async def test_run_agent_rolls_progress_bubble_before_platform_limit(monkeypatch, tmp_path):
    """Tool progress should start a second editable bubble before Telegram's limit.

    Regression: once the first progress bubble grew past the platform limit,
    the gateway kept trying to edit that same oversized full transcript.  The
    Telegram adapter then split-and-sent a fresh continuation on every update,
    causing a noisy trail of one-line messages instead of a new editable bubble.
    """
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        ManyProgressLinesAgent,
        session_id="sess-progress-overflow-rollover",
        config_data={
            "display": {
                "tool_progress": "all",
                "interim_assistant_messages": False,
                "tool_preview_length": 60,
            }
        },
        adapter_cls=SmallLimitProgressAdapter,
    )

    assert result["final_response"] == "done"
    assert isinstance(adapter, SmallLimitProgressAdapter)
    assert len(adapter.sent) >= 2, "expected a fresh progress bubble after the first filled"
    assert adapter.oversized_sends == []
    assert adapter.oversized_edits == []
    all_bubbles = [call["content"] for call in adapter.sent + adapter.edits]
    assert all(len(text) <= adapter.MAX_MESSAGE_LENGTH for text in all_bubbles)


@pytest.mark.asyncio
async def test_run_agent_surfaces_real_interim_commentary(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary",
        config_data={"display": {"interim_assistant_messages": True}},
    )

    assert result.get("already_sent") is not True
    assert any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_surfaces_interim_commentary_by_default(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-default-on",
    )

    assert any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_suppresses_interim_commentary_when_disabled(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-disabled",
        config_data={"display": {"interim_assistant_messages": False}},
    )

    assert result.get("already_sent") is not True
    assert not any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_tool_progress_does_not_control_interim_commentary(monkeypatch, tmp_path):
    """tool_progress=all with interim_assistant_messages=false should not surface commentary."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-tool-progress",
        config_data={"display": {"tool_progress": "all", "interim_assistant_messages": False}},
    )

    assert result.get("already_sent") is not True
    assert not any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_streaming_does_not_enable_completed_interim_commentary(
    monkeypatch, tmp_path
):
    """Streaming alone with interim_assistant_messages=false should not surface commentary."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-streaming",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "streaming": {"enabled": True},
        },
    )

    assert result.get("already_sent") is True
    assert not any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_display_streaming_does_not_enable_gateway_streaming(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-display-streaming-cli-only",
        config_data={
            "display": {
                "streaming": True,
                "interim_assistant_messages": True,
            },
            "streaming": {"enabled": False},
        },
    )

    assert result.get("already_sent") is not True
    assert adapter.edits == []
    assert [call["content"] for call in adapter.sent] == ["I'll inspect the repo first."]


@pytest.mark.asyncio
async def test_run_agent_interim_commentary_works_with_tool_progress_off(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-explicit-on",
        config_data={
            "display": {
                "tool_progress": "off",
                "interim_assistant_messages": True,
            },
        },
    )

    assert result.get("already_sent") is not True
    assert any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_bluebubbles_uses_commentary_send_path_for_quick_replies(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-bluebubbles-commentary",
        config_data={"display": {"interim_assistant_messages": True}},
        platform=Platform.BLUEBUBBLES,
        chat_id="iMessage;-;user@example.com",
        chat_type="dm",
        thread_id=None,
        adapter_cls=NonEditingProgressCaptureAdapter,
    )

    assert result.get("already_sent") is not True
    assert [call["content"] for call in adapter.sent] == ["I'll inspect the repo first."]
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_run_agent_previewed_final_marks_already_sent(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        PreviewedResponseAgent,
        session_id="sess-previewed",
        config_data={"display": {"interim_assistant_messages": True}},
    )

    assert result.get("already_sent") is True
    assert [call["content"] for call in adapter.sent] == ["You're welcome."]


@pytest.mark.asyncio
async def test_run_agent_matrix_streaming_omits_cursor(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        StreamingRefineAgent,
        session_id="sess-matrix-streaming",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "streaming": {"enabled": True, "edit_interval": 0.01, "buffer_threshold": 1},
        },
        platform=Platform.MATRIX,
        chat_id="!room:matrix.example.org",
        chat_type="group",
        thread_id="$thread",
    )

    assert result.get("already_sent") is True
    all_text = [call["content"] for call in adapter.sent] + [call["content"] for call in adapter.edits]
    assert all_text, "expected streamed Matrix content to be sent or edited"
    assert all("▉" not in text for text in all_text)
    assert any("Continuing to refine:" in text for text in all_text)


@pytest.mark.asyncio
async def test_run_agent_queued_message_does_not_treat_commentary_as_final(monkeypatch, tmp_path):
    QueuedCommentaryAgent.calls = 0
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedCommentaryAgent,
        session_id="sess-queued-commentary",
        pending_text="queued follow-up",
        config_data={"display": {"interim_assistant_messages": True}},
    )

    sent_texts = [call["content"] for call in adapter.sent]
    assert result["final_response"] == "final response 2"
    assert "I'll inspect the repo first." in sent_texts
    assert "final response 1" in sent_texts


@pytest.mark.asyncio
async def test_recursive_private_queue_routes_interim_to_private_workspace(
    monkeypatch, tmp_path
):
    EveryTurnCommentaryAgent.calls = 0
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        chat_type="group",
        thread_id="root-1",
    )
    original = MessageEvent(
        text="active public turn",
        source=source,
        message_id="root-1",
        expects_reply=True,
    )
    queued = MessageEvent(
        text="private queued turn",
        source=source,
        message_id="queued-1",
        expects_reply=True,
        private_reply_user_id="U_PRIVATE",
        platform_team_id="T_SECONDARY",
    )

    adapter, _result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        EveryTurnCommentaryAgent,
        session_id="sess-private-queue-route",
        platform=Platform.SLACK,
        chat_id="C1",
        chat_type="group",
        thread_id="root-1",
        reply_event=original,
        pending_event=queued,
        config_data={"display": {"interim_assistant_messages": True}},
    )

    second_commentary = next(
        call for call in adapter.sent if call["content"] == "private commentary 2"
    )
    assert second_commentary["metadata"] == {
        "thread_id": "root-1",
        "private_reply_user_id": "U_PRIVATE",
        "team_id": "T_SECONDARY",
    }


@pytest.mark.asyncio
async def test_queued_turn_does_not_rewrite_first_response_targets(monkeypatch, tmp_path):
    QueuedCommentaryAgent.calls = 0
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    original = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="original-1",
        expects_reply=True,
    )
    deferred = MessageEvent(
        text="clarification",
        message_type=MessageType.TEXT,
        source=source,
        message_id="clarify-1",
        expects_reply=True,
    )
    queued = MessageEvent(
        text="queued follow-up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-1",
        expects_reply=True,
    )
    original.delivery_state.final_response_events.extend([original, deferred])

    _adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedCommentaryAgent,
        session_id="sess-queued-reply-targets",
        reply_event=original,
        pending_event=queued,
    )

    assert original.delivery_state.reply_delivered is True
    assert deferred.delivery_state.reply_delivered is True
    assert original.delivery_state.final_response_events == []
    assert result["_final_reply_events"] == [queued]


@pytest.mark.asyncio
async def test_queue_accepted_before_drain_gets_a_terminal_reply(monkeypatch, tmp_path):
    QueuedCommentaryAgent.calls = 0
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    queued = MessageEvent(
        text="queued before restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-before-drain",
        expects_reply=True,
    )

    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedCommentaryAgent,
        session_id="sess-queue-before-drain",
        pending_event=queued,
        draining=True,
    )

    assert any("resend" in call["content"] for call in adapter.sent)
    assert queued.delivery_state.reply_delivered is True
    assert any(item is queued for item in result["_reply_events"])


@pytest.mark.asyncio
async def test_drain_rejects_every_accepted_queued_turn(monkeypatch, tmp_path):
    QueuedCommentaryAgent.calls = 0
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    queued = [
        MessageEvent(
            text=f"queued {index}",
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"queued-{index}",
            expects_reply=True,
        )
        for index in range(3)
    ]

    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedCommentaryAgent,
        session_id="sess-all-queue-before-drain",
        pending_event=queued[0],
        overflow_events=queued[1:],
        draining=True,
    )

    rejections = [call for call in adapter.sent if "resend" in call["content"]]
    assert len(rejections) == 3
    assert all(event.delivery_state.reply_delivered for event in queued)
    assert all(any(item is event for item in result["_reply_events"]) for event in queued)


@pytest.mark.asyncio
async def test_failed_agent_drain_rejects_every_accepted_queued_turn(
    monkeypatch, tmp_path
):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    queued = [
        MessageEvent(
            text=f"queued {index}",
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"failed-queued-{index}",
            expects_reply=True,
        )
        for index in range(3)
    ]

    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    runner._draining = True
    session_key = "agent:main:telegram:group:-1001:17585"
    adapter._pending_messages[session_key] = queued[0]
    runner._queued_events = {session_key: queued[1:]}

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = RaisingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )

    with pytest.raises(RuntimeError, match="agent failed"):
        await runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-failed-drain",
            session_key=session_key,
        )

    rejections = [call for call in adapter.sent if "resend" in call["content"]]
    assert len(rejections) == 3
    assert all(event.delivery_state.reply_delivered for event in queued)
    assert session_key not in adapter._pending_messages
    assert session_key not in runner._queued_events


@pytest.mark.asyncio
async def test_failed_drain_completes_each_queued_delivery_outcome():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    queued = [
        MessageEvent(
            text=f"queued {index}",
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"outcome-{index}",
            expects_reply=True,
        )
        for index in range(3)
    ]
    merged = MessageEvent(
        text="merged",
        message_type=MessageType.TEXT,
        source=source,
        message_id="outcome-merged",
        expects_reply=True,
    )
    queued[0].delivery_state.merged_events.append(merged)
    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    adapter._send_with_retry = AsyncMock(
        side_effect=[
            SendResult(success=True, message_id="ok"),
            SendResult(success=False, error="rejected"),
            RuntimeError("send failed"),
        ]
    )
    adapter.on_processing_complete = AsyncMock()
    runner = _make_runner(adapter)
    runner._draining = True
    runner._restart_requested = True
    session_key = "agent:main:telegram:group:-1001:17585"
    adapter._pending_messages[session_key] = queued[0]
    runner._queued_events = {session_key: queued[1:]}

    await runner._reject_queued_events_after_failed_drain(session_key, source)

    outcomes = [call.args[1] for call in adapter.on_processing_complete.await_args_list]
    assert outcomes == [
        ProcessingOutcome.SUCCESS,
        ProcessingOutcome.SUCCESS,
        ProcessingOutcome.FAILURE,
        ProcessingOutcome.FAILURE,
    ]
    assert queued[0].delivery_state.reply_delivered is True
    assert merged.delivery_state.reply_delivered is True
    assert queued[1].delivery_state.reply_delivered is False
    assert queued[2].delivery_state.reply_delivered is False


def test_pending_event_order_accepts_mixed_timezone_timestamps():
    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )
    aware = MessageEvent(
        text="aware first",
        message_type=MessageType.TEXT,
        source=source,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    naive = MessageEvent(
        text="naive second",
        message_type=MessageType.TEXT,
        source=source,
        timestamp=datetime(2026, 1, 1, 0, 0, 1),
    )

    first = runner._merge_pending_events_by_arrival(
        "session",
        adapter,
        naive,
        aware,
    )

    assert first is aware
    assert adapter._pending_messages["session"] is naive


@pytest.mark.asyncio
async def test_repeated_run_failures_promote_every_overflow_turn(monkeypatch, tmp_path):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    queued = [
        MessageEvent(
            text=f"queued {index}",
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"failure-chain-{index}",
            expects_reply=True,
        )
        for index in range(2)
    ]
    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    session_key = "agent:main:telegram:group:-1001:17585"
    adapter._pending_messages[session_key] = queued[0]
    runner._queued_events = {session_key: [queued[1]]}

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = RaisingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )

    for expected in queued:
        with pytest.raises(RuntimeError, match="agent failed"):
            await runner._run_agent(
                message="hello",
                context_prompt="",
                history=[],
                source=source,
                session_id="sess-repeated-failure",
                session_key=session_key,
            )
        assert adapter._pending_messages.pop(session_key) is expected

    assert session_key not in runner._queued_events


@pytest.mark.asyncio
async def test_early_return_preserves_a_late_steer_as_the_next_turn(
    monkeypatch,
    tmp_path,
):
    EarlyReturnSteerAgent.calls = 0

    _adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        EarlyReturnSteerAgent,
        session_id="sess-early-return-steer",
    )

    assert EarlyReturnSteerAgent.calls == 2
    assert result["final_response"] == "final response 2"


@pytest.mark.asyncio
async def test_empty_early_return_preserves_a_late_steer(monkeypatch, tmp_path):
    EmptyEarlyReturnSteerAgent.calls = 0
    EmptyEarlyReturnSteerAgent.messages = []

    _adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        EmptyEarlyReturnSteerAgent,
        session_id="sess-empty-early-return-steer",
    )

    assert EmptyEarlyReturnSteerAgent.messages == ["hello", "late steer"]
    assert result["final_response"] == "steer response"


@pytest.mark.asyncio
async def test_late_steer_runs_before_later_queued_turn(monkeypatch, tmp_path):
    EarlyReturnSteerAgent.calls = 0
    EarlyReturnSteerAgent.messages = []
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    queued = MessageEvent(
        text="queued follow-up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-after-steer",
        expects_reply=True,
        timestamp=datetime(2026, 1, 1, 0, 0, 1),
    )
    steer = MessageEvent(
        text="/steer late steer",
        message_type=MessageType.TEXT,
        source=source,
        message_id="late-steer",
        expects_reply=True,
        timestamp=datetime(2026, 1, 1, 0, 0, 0),
    )

    _adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        EarlyReturnSteerAgent,
        session_id="sess-steer-before-queue",
        pending_event=queued,
        steer_events=[("late steer", steer)],
    )

    assert EarlyReturnSteerAgent.messages == ["hello", "late steer", "queued follow-up"]
    assert result["final_response"] == "final response 3"


@pytest.mark.asyncio
async def test_late_steer_runs_after_earlier_queued_turn(monkeypatch, tmp_path):
    EarlyReturnSteerAgent.calls = 0
    EarlyReturnSteerAgent.messages = []
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    queued = MessageEvent(
        text="queued follow-up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-before-steer",
        expects_reply=True,
        timestamp=datetime(2026, 1, 1, 0, 0, 0),
    )
    steer = MessageEvent(
        text="/steer late steer",
        message_type=MessageType.TEXT,
        source=source,
        message_id="late-steer",
        expects_reply=True,
        timestamp=datetime(2026, 1, 1, 0, 0, 1),
    )

    _adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        EarlyReturnSteerAgent,
        session_id="sess-queue-before-steer",
        pending_event=queued,
        steer_events=[("late steer", steer)],
    )

    assert EarlyReturnSteerAgent.messages == ["hello", "queued follow-up", "late steer"]
    assert result["final_response"] == "final response 3"


@pytest.mark.asyncio
async def test_recursion_limit_preserves_current_turn_before_staged_queue(
    monkeypatch, tmp_path
):
    EarlyReturnSteerAgent.calls = 0
    EarlyReturnSteerAgent.messages = []
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    first = MessageEvent(
        text="first queued",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-first",
    )
    second = MessageEvent(
        text="second queued",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-second",
    )

    adapter, _result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        EarlyReturnSteerAgent,
        session_id="sess-depth-queue",
        pending_event=first,
        overflow_events=[second],
        interrupt_depth=100,
    )

    session_key = "agent:main:telegram:group:-1001:17585"
    assert adapter._pending_messages[session_key] is first
    assert adapter._test_runner._queued_events[session_key] == [second]


@pytest.mark.asyncio
async def test_media_only_result_is_not_replaced_by_empty_response(monkeypatch, tmp_path):
    _adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        MediaOnlyAgent,
        session_id="sess-media-only",
    )

    assert result["final_response"] == "MEDIA:/tmp/report.pdf"


@pytest.mark.asyncio
async def test_queued_followup_delivers_first_turn_tool_media(monkeypatch, tmp_path):
    QueuedMediaAgent.calls = 0
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    owner = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="media-owner",
        expects_reply=True,
    )
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedMediaAgent,
        session_id="sess-queued-media",
        pending_text="queued follow-up",
        reply_event=owner,
    )

    assert result["final_response"] == "queued response"
    assert any(
        call["content"] == "📎 File: /tmp/queued-report.pdf"
        for call in adapter.sent
    )
    assert all("MEDIA:" not in call["content"] for call in adapter.sent)
    assert owner.delivery_state.reply_attempted is True
    assert owner.delivery_state.reply_delivered is True


@pytest.mark.asyncio
async def test_queued_followup_delivers_first_turn_markdown_image(monkeypatch, tmp_path):
    QueuedMarkdownImageAgent.calls = 0
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    owner = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="image-owner",
        expects_reply=True,
    )
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedMarkdownImageAgent,
        session_id="sess-queued-markdown-image",
        pending_text="queued follow-up",
        reply_event=owner,
        adapter_cls=ImageCaptureAdapter,
    )

    assert result["final_response"] == "queued response"
    assert adapter.image_batches == [
        [("https://example.com/chart.png", "chart")]
    ]
    assert owner.delivery_state.reply_delivered is True
    assert owner.delivery_state.reply_failed is False


@pytest.mark.asyncio
async def test_queued_media_and_notice_failure_mark_owner_failed(monkeypatch, tmp_path):
    QueuedFailedMediaAgent.calls = 0
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    owner = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="failed-media-owner",
        expects_reply=True,
    )

    _adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedFailedMediaAgent,
        session_id="sess-queued-failed-media",
        pending_text="queued follow-up",
        reply_event=owner,
        adapter_cls=FailedMediaNoticeAdapter,
    )

    assert result["final_response"] == "queued response"
    assert owner.delivery_state.reply_delivered is True
    assert owner.delivery_state.reply_failed is True


@pytest.mark.asyncio
async def test_queued_image_batch_failure_uses_generic_batch_name(monkeypatch, tmp_path):
    QueuedFailedImageBatchAgent.calls = 0
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    owner = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="failed-image-batch-owner",
        expects_reply=True,
    )

    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedFailedImageBatchAgent,
        session_id="sess-queued-failed-image-batch",
        pending_text="queued follow-up",
        reply_event=owner,
        adapter_cls=FailedImageBatchAdapter,
    )

    assert result["final_response"] == "queued response"
    notice = next(item["content"] for item in adapter.sent if "was not attached" in item["content"])
    assert "2-image batch" in notice
    assert "first.png" not in notice
    assert owner.delivery_state.reply_failed is True
    assert owner.delivery_state.failure_notice_delivered is True


@pytest.mark.asyncio
async def test_recursive_queued_failure_preserves_each_reply_owner(monkeypatch, tmp_path):
    SecondCallRaisingAgent.calls = 0
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    original = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="original-owner",
        expects_reply=True,
    )
    queued = MessageEvent(
        text="queued",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-owner",
        expects_reply=True,
    )

    with pytest.raises(RuntimeError, match="queued agent failed"):
        await _run_with_agent(
            monkeypatch,
            tmp_path,
            SecondCallRaisingAgent,
            session_id="sess-recursive-failure-owner",
            pending_event=queued,
            reply_event=original,
        )

    assert queued in original.delivery_state.completion_events


@pytest.mark.asyncio
@pytest.mark.parametrize("with_queued_owner", [False, True])
async def test_real_handler_propagates_queued_owners_and_streamed_delivery(
    monkeypatch,
    with_queued_owner,
):
    """Exercise the production bridge from runner metadata to event state."""
    from gateway.run import GatewayRunner

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    event = MessageEvent(
        text="original",
        message_type=MessageType.TEXT,
        source=source,
        message_id="original-1",
        expects_reply=True,
    )
    queued = MessageEvent(
        text="queued",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-1",
        expects_reply=True,
    )
    session_key = build_session_key(source)
    created = datetime.now()
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-real-handler",
        created_at=created,
        updated_at=created + timedelta(seconds=1),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )

    adapter = ProgressCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")
        }
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_db = None
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._show_reasoning = False
    runner._recover_telegram_topic_thread_id = MagicMock(return_value=None)
    runner._cache_session_source = MagicMock()
    runner._is_telegram_topic_lane = MagicMock(return_value=False)
    runner._set_session_env = MagicMock(return_value=())
    runner._clear_session_env = MagicMock()
    runner._prepare_inbound_message_text = AsyncMock(return_value=event.text)
    runner._bind_adapter_run_generation = MagicMock()
    runner._is_session_run_current = MagicMock(return_value=True)
    runner._clear_restart_failure_count = MagicMock()
    runner._should_send_voice_reply = MagicMock(return_value=False)
    runner._deliver_media_from_response = AsyncMock()
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "streamed answer",
            "messages": [],
            "api_calls": 1,
            "already_sent": True,
            "failed": False,
            "_reply_events": [queued] if with_queued_owner else [],
            "_final_reply_events": [queued] if with_queued_owner else [],
        }
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    response = await runner._handle_message_with_agent(
        event,
        source,
        session_key,
        run_generation=1,
    )

    assert response is None
    if with_queued_owner:
        assert event.delivery_state.reply_delivered is False
        assert event.delivery_state.reply_failed is False
        assert queued.delivery_state.reply_delivered is True
        assert queued.delivery_state.reply_failed is False
        assert queued in event.delivery_state.completion_events
        assert queued in event.delivery_state.final_response_events
    else:
        assert event.delivery_state.reply_delivered is True
        assert event.delivery_state.reply_failed is False
        assert queued.delivery_state.reply_delivered is False


@pytest.mark.asyncio
async def test_streamed_media_and_notice_failure_marks_reply_failed(monkeypatch):
    """A streamed text fragment cannot hide a failed attachment outcome."""
    from gateway.run import GatewayRunner

    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="group",
        thread_id="thread-1",
    )
    event = MessageEvent(
        text="original",
        message_type=MessageType.TEXT,
        source=source,
        message_id="original-1",
        expects_reply=True,
    )
    session_key = build_session_key(source)
    created = datetime.now()
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-streamed-media-failure",
        created_at=created,
        updated_at=created + timedelta(seconds=1),
        platform=Platform.SLACK,
        chat_type="group",
    )

    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.SLACK: adapter}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_db = None
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._show_reasoning = False
    runner._recover_telegram_topic_thread_id = MagicMock(return_value=None)
    runner._cache_session_source = MagicMock()
    runner._is_telegram_topic_lane = MagicMock(return_value=False)
    runner._set_session_env = MagicMock(return_value=())
    runner._clear_session_env = MagicMock()
    runner._prepare_inbound_message_text = AsyncMock(return_value=event.text)
    runner._bind_adapter_run_generation = MagicMock()
    runner._is_session_run_current = MagicMock(return_value=True)
    runner._clear_restart_failure_count = MagicMock()
    runner._should_send_voice_reply = MagicMock(return_value=False)
    runner._deliver_media_from_response = AsyncMock(return_value=False)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Report attached.\nMEDIA:/tmp/report.pdf",
            "messages": [],
            "api_calls": 1,
            "already_sent": True,
            "failed": False,
        }
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    response = await runner._handle_message_with_agent(
        event,
        source,
        session_key,
        run_generation=1,
    )

    assert response is None
    assert event.delivery_state.reply_delivered is True
    assert event.delivery_state.reply_failed is True


@pytest.mark.asyncio
async def test_run_agent_defers_background_review_notification_until_release(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        BackgroundReviewAgent,
        session_id="sess-bg-review-order",
        config_data={"display": {"interim_assistant_messages": True}},
    )

    assert result["final_response"] == "done"
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_base_processing_releases_post_delivery_callback_after_main_send():
    """Post-delivery callbacks on the adapter fire after the main response."""
    adapter = ProgressCaptureAdapter()

    async def _handler(event):
        return "done"

    adapter.set_message_handler(_handler)

    released = []

    def _post_delivery_cb():
        released.append(True)
        adapter.sent.append(
            {
                "chat_id": "bg-review",
                "content": "💾 Skill 'prospect-scanner' created.",
                "reply_to": None,
                "metadata": None,
            }
        )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )
    session_key = "agent:main:telegram:group:-1001:17585"
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._post_delivery_callbacks[session_key] = _post_delivery_cb

    await adapter._process_message_background(event, session_key)

    sent_texts = [call["content"] for call in adapter.sent]
    assert sent_texts == ["done", "💾 Skill 'prospect-scanner' created."]
    assert released == [True]


@pytest.mark.asyncio
async def test_run_agent_drops_tool_progress_after_generation_invalidation(monkeypatch, tmp_path):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.dump({"display": {"tool_progress": "all"}}),
        encoding="utf-8",
    )

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = DelayedProgressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 - register terminal tool metadata

    adapter = ProgressCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="dm-1",
        chat_type="dm",
        thread_id=None,
    )
    session_key = "agent:main:discord:dm:dm-1"
    runner._session_run_generation[session_key] = 1

    original_send = adapter.send
    invalidated = {"done": False}

    async def send_and_invalidate(chat_id, content, reply_to=None, metadata=None):
        result = await original_send(chat_id, content, reply_to=reply_to, metadata=metadata)
        if "first command" in content and not invalidated["done"]:
            invalidated["done"] = True
            runner._invalidate_session_run_generation(session_key, reason="test_stop")
        return result

    adapter.send = send_and_invalidate

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-progress-stop",
        session_key=session_key,
        run_generation=1,
    )

    all_progress_text = " ".join(call["content"] for call in adapter.sent)
    all_progress_text += " ".join(call["content"] for call in adapter.edits)
    assert result["final_response"] == "done"
    assert 'first command' in all_progress_text
    assert 'second command' not in all_progress_text


@pytest.mark.asyncio
async def test_run_agent_drops_interim_commentary_after_generation_invalidation(monkeypatch, tmp_path):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.dump({"display": {"tool_progress": "off", "interim_assistant_messages": True}}),
        encoding="utf-8",
    )

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = DelayedInterimAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="dm-2",
        chat_type="dm",
        thread_id=None,
    )
    session_key = "agent:main:discord:dm:dm-2"
    runner._session_run_generation[session_key] = 1

    original_send = adapter.send
    invalidated = {"done": False}

    async def send_and_invalidate(chat_id, content, reply_to=None, metadata=None):
        result = await original_send(chat_id, content, reply_to=reply_to, metadata=metadata)
        if content == "first interim" and not invalidated["done"]:
            invalidated["done"] = True
            runner._invalidate_session_run_generation(session_key, reason="test_stop")
        return result

    adapter.send = send_and_invalidate

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-commentary-stop",
        session_key=session_key,
        run_generation=1,
    )

    sent_texts = [call["content"] for call in adapter.sent]
    assert result["final_response"] == "done"
    assert "first interim" in sent_texts
    assert "second interim" not in sent_texts


@pytest.mark.asyncio
async def test_keep_typing_stops_immediately_when_interrupt_event_is_set():
    adapter = ProgressCaptureAdapter(platform=Platform.DISCORD)
    stop_event = asyncio.Event()

    task = asyncio.create_task(
        adapter._keep_typing(
            "dm-typing-stop",
            interval=30.0,
            stop_event=stop_event,
        )
    )
    await asyncio.sleep(0.05)
    stop_event.set()
    await asyncio.wait_for(task, timeout=0.5)

    normal_typing_calls = [
        call for call in adapter.typing if call.get("metadata") != {"stopped": True}
    ]
    stopped_calls = [
        call for call in adapter.typing if call.get("metadata") == {"stopped": True}
    ]
    assert len(normal_typing_calls) == 1
    assert len(stopped_calls) == 1


@pytest.mark.asyncio
async def test_verbose_mode_does_not_truncate_args_by_default(monkeypatch, tmp_path):
    """Verbose mode with default tool_preview_length (0) should NOT truncate args.

    Previously, verbose mode capped args at 200 chars when tool_preview_length
    was 0 (default).  The user explicitly opted into verbose — show full detail.
    """
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VerboseAgent,
        session_id="sess-verbose-no-truncate",
        config_data={"display": {"tool_progress": "verbose", "tool_preview_length": 0}},
    )

    assert result["final_response"] == "done"
    # The full 300-char 'x' string should be present, not truncated to 200
    all_content = " ".join(call["content"] for call in adapter.sent)
    all_content += " ".join(call["content"] for call in adapter.edits)
    assert VerboseAgent.LONG_CODE in all_content


@pytest.mark.asyncio
async def test_verbose_mode_respects_explicit_tool_preview_length(monkeypatch, tmp_path):
    """When tool_preview_length is set to a positive value, verbose truncates to that."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        VerboseAgent,
        session_id="sess-verbose-explicit-cap",
        config_data={"display": {"tool_progress": "verbose", "tool_preview_length": 50}},
    )

    assert result["final_response"] == "done"
    all_content = " ".join(call["content"] for call in adapter.sent)
    all_content += " ".join(call["content"] for call in adapter.edits)
    # Should be truncated — full 300-char string NOT present
    assert VerboseAgent.LONG_CODE not in all_content
    # But should still contain the truncated portion with "..."
    assert "..." in all_content
