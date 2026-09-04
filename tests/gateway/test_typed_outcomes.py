"""Typed, content-minimized gateway outcome events."""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageDeliveryState,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.config import PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.typed_outcomes import (
    DeliveryCompleteEvent,
    TurnTerminalEvent,
    build_delivery_complete_event,
    build_turn_terminal_event,
)
from hermes_cli.plugins import VALID_HOOKS


def _event(*, text: str = "private request", expects_reply: bool = True) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        message_id="1788460000.123456",
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="thread",
            user_id="U123",
            thread_id="1788450000.000001",
        ),
        platform_team_id="T123",
        expects_reply=expects_reply,
    )


def test_typed_outcome_hooks_are_public_plugin_hooks():
    assert "turn_terminal" in VALID_HOOKS
    assert "delivery_complete" in VALID_HOOKS


def test_turn_terminal_event_is_stable_and_omits_request_body():
    first = build_turn_terminal_event(
        event=_event(text="top secret request"),
        session_id="session-1",
        run_generation=7,
        outcome="failed",
        failure_class="provider_rate_limit",
        error=RuntimeError("api-key=secret-value"),
        api_calls=4,
        tool_names=("slack__get_thread_replies",),
    )
    second = build_turn_terminal_event(
        event=_event(text="different retry body"),
        session_id="session-1",
        run_generation=7,
        outcome="failed",
        failure_class="provider_rate_limit",
        error=RuntimeError("api-key=secret-value"),
        api_calls=4,
        tool_names=("slack__get_thread_replies",),
    )

    assert isinstance(first, TurnTerminalEvent)
    assert first.event_id == second.event_id
    assert first.schema_version == 1
    assert first.platform == "slack"
    assert first.team_id == "T123"
    assert first.chat_id == "C123"
    assert first.thread_id == "1788450000.000001"
    assert first.message_id == "1788460000.123456"
    assert first.user_id == "U123"
    assert first.outcome == "failed"
    assert first.failure_class == "provider_rate_limit"
    assert first.api_calls == 4
    assert first.tool_names == ("slack__get_thread_replies",)
    encoded = repr(first)
    assert "top secret request" not in encoded
    assert "different retry body" not in encoded
    assert "secret-value" not in encoded


def test_turn_terminal_event_omits_arbitrary_private_error_text():
    private_text = "PRIVATE CUSTOMER QUESTION: acquisition terms for Acme"
    outcome = build_turn_terminal_event(
        event=_event(),
        session_id="session-1",
        run_generation=7,
        outcome="failed",
        failure_class="unhandled_exception",
        error=RuntimeError(private_text),
    )

    assert outcome.error_type == "RuntimeError"
    assert outcome.error_detail is None
    assert private_text not in repr(outcome)


def test_delivery_event_distinguishes_empty_required_reply():
    event = _event()
    outcome = build_delivery_complete_event(
        event=event,
        outcome=ProcessingOutcome.FAILURE,
    )

    assert isinstance(outcome, DeliveryCompleteEvent)
    assert outcome.failure_class == "empty_required_reply"
    assert outcome.reply_attempted is False
    assert outcome.reply_delivered is False
    assert outcome.event_id.startswith("delivery-complete:v1:")


def test_delivery_event_records_confirmed_message_ids_without_content():
    event = _event(text="do not persist me")
    event.delivery_state = MessageDeliveryState(
        reply_attempted=True,
        reply_delivered=True,
        reply_message_ids=["1788460001.000002", "1788460002.000003"],
    )
    outcome = build_delivery_complete_event(
        event=event,
        outcome=ProcessingOutcome.SUCCESS,
    )

    assert outcome.failure_class is None
    assert outcome.reply_message_ids == (
        "1788460001.000002",
        "1788460002.000003",
    )
    assert "do not persist me" not in repr(outcome)


def test_delivery_event_is_stable_across_transport_retry():
    event = _event()
    event.delivery_state.reply_attempted = True
    event.delivery_state.reply_failed = True
    first = build_delivery_complete_event(
        event=event,
        outcome=ProcessingOutcome.FAILURE,
    )
    second = build_delivery_complete_event(
        event=event,
        outcome=ProcessingOutcome.FAILURE,
    )

    assert first.event_id == second.event_id
    assert first.failure_class == "delivery_failure"


class _Adapter(BasePlatformAdapter):
    async def connect(self):
        return None

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, **kwargs):
        return SendResult(success=True, message_id="1788460001.000002")

    async def get_chat_info(self, chat_id):
        return {}


@pytest.mark.asyncio
async def test_platform_delivery_emits_one_typed_hook(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        "gateway.typed_outcomes.emit_plugin_event",
        lambda hook_name, event: emitted.append((hook_name, event)),
    )
    adapter = _Adapter(PlatformConfig(enabled=True, token="t"), Platform.SLACK)
    adapter._message_handler = AsyncMock(return_value="substantive answer")
    event = _event()

    await adapter._process_message_background(event, "slack:T123:C123:U123")

    delivery_events = [item for item in emitted if item[0] == "delivery_complete"]
    assert len(delivery_events) == 1
    typed = delivery_events[0][1]
    assert typed.outcome == "success"
    assert typed.reply_delivered is True
    assert typed.reply_message_ids == ("1788460001.000002",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "clarify_pending", "send_success"),
    [
        ("/status", False, True),
        ("/status", False, False),
        ("clarification", True, True),
        ("clarification", True, False),
    ],
)
async def test_direct_active_session_paths_emit_delivery_complete(
    monkeypatch,
    text,
    clarify_pending,
    send_success,
):
    emitted = []
    monkeypatch.setattr(
        "gateway.typed_outcomes.emit_plugin_event",
        lambda hook_name, event: emitted.append((hook_name, event)),
    )
    monkeypatch.setattr(
        "tools.clarify_gateway.get_pending_for_session",
        lambda _session_key: object() if clarify_pending else None,
    )
    adapter = _Adapter(PlatformConfig(enabled=True, token="t"), Platform.SLACK)
    adapter._message_handler = AsyncMock(return_value="substantive answer")
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(
            success=send_success,
            message_id="1788460001.000002" if send_success else None,
        )
    )
    event = _event(text=text)
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    delivery_events = [item for item in emitted if item[0] == "delivery_complete"]
    assert len(delivery_events) == 1
    assert delivery_events[0][1].outcome == (
        "success" if send_success else "failure"
    )


@pytest.mark.asyncio
async def test_post_agent_failure_emits_only_one_failed_terminal_event(monkeypatch):
    source = _event().source
    event = _event()
    session_key = build_session_key(source)
    created = datetime.now()
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-post-agent-failure",
        created_at=created,
        updated_at=created + timedelta(seconds=1),
        platform=Platform.SLACK,
        chat_type="group",
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.SLACK: _Adapter(PlatformConfig(), Platform.SLACK)}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
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
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "answer",
            "messages": [],
            "api_calls": 1,
            "failed": False,
        }
    )

    async def emit_hook(name, _payload):
        if name == "agent:end":
            raise RuntimeError("PRIVATE CUSTOMER QUESTION: acquisition terms")

    runner.hooks = SimpleNamespace(emit=emit_hook, loaded_hooks=False)
    emitted = []
    monkeypatch.setattr(
        "gateway.typed_outcomes.emit_plugin_event",
        lambda hook_name, outcome: emitted.append((hook_name, outcome)),
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    response = await runner._handle_message_with_agent(
        event,
        source,
        session_key,
        run_generation=1,
    )

    terminal_events = [item for item in emitted if item[0] == "turn_terminal"]
    assert len(terminal_events) == 1
    assert terminal_events[0][1].outcome == "failed"
    assert terminal_events[0][1].failure_class == "unhandled_exception"
    assert terminal_events[0][1].error_detail is None
    assert "Sorry, I encountered an error" in response
