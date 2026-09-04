"""Typed, content-minimized gateway outcome events."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageDeliveryState,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.config import PlatformConfig
from gateway.session import SessionSource
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
