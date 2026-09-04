"""Typed, content-minimized outcomes for gateway turns and deliveries.

Contract
--------
``turn_terminal`` fires once when an agent turn reaches a terminal state.
``delivery_complete`` fires once when the platform adapter has terminal
delivery evidence. Event identifiers are stable across transport retries.

These events contain coordinates, hashes, classifications, and sanitized
errors. They never contain the inbound request body or response body.
Plugin failures remain isolated by :func:`hermes_cli.plugins.invoke_hook`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
def _source_value(event: Any, name: str) -> str:
    source = getattr(event, "source", None)
    value = getattr(source, name, None)
    if name == "platform":
        value = getattr(value, "value", value)
    return str(value or "")


def _stable_id(kind: str, event: Any, *parts: object) -> str:
    source = getattr(event, "source", None)
    identity = {
        "platform": _source_value(event, "platform"),
        "team_id": str(getattr(event, "platform_team_id", None) or ""),
        "chat_id": _source_value(event, "chat_id"),
        "thread_id": _source_value(event, "thread_id"),
        "message_id": str(
            getattr(event, "message_id", None)
            or getattr(source, "message_id", None)
            or ""
        ),
        "user_id": _source_value(event, "user_id"),
        "parts": [str(part or "") for part in parts],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{kind}:v{SCHEMA_VERSION}:{digest}"


def _safe_error(error: BaseException | str | None) -> tuple[str | None, str | None]:
    if error is None:
        return None, None
    error_type = type(error).__name__ if isinstance(error, BaseException) else "Error"
    # Exception messages can contain the inbound request, provider response,
    # credentials, or customer data.  A best-effort redactor cannot prove that
    # arbitrary prose is safe.  Keep the schema field for compatibility, but
    # publish only the controlled exception class.
    return error_type, None


def _coordinates(event: Any) -> dict[str, Any]:
    source = getattr(event, "source", None)
    return {
        "platform": _source_value(event, "platform"),
        "team_id": str(getattr(event, "platform_team_id", None) or ""),
        "chat_id": _source_value(event, "chat_id"),
        "thread_id": _source_value(event, "thread_id"),
        "message_id": str(
            getattr(event, "message_id", None)
            or getattr(source, "message_id", None)
            or ""
        ),
        "user_id": _source_value(event, "user_id"),
    }


@dataclass(frozen=True)
class TurnTerminalEvent:
    schema_version: int
    event_id: str
    occurred_at: str
    platform: str
    team_id: str
    chat_id: str
    thread_id: str
    message_id: str
    user_id: str
    session_id: str
    run_generation: int
    outcome: str
    failure_class: str | None
    error_type: str | None
    error_detail: str | None
    api_calls: int
    tool_names: tuple[str, ...]
    response_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeliveryCompleteEvent:
    schema_version: int
    event_id: str
    occurred_at: str
    platform: str
    team_id: str
    chat_id: str
    thread_id: str
    message_id: str
    user_id: str
    outcome: str
    failure_class: str | None
    expects_reply: bool
    reply_attempted: bool
    reply_delivered: bool
    reply_failed: bool
    failure_notice_delivered: bool
    reply_message_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_turn_terminal_event(
    *,
    event: Any,
    session_id: str,
    run_generation: int,
    outcome: str,
    failure_class: str | None = None,
    error: BaseException | str | None = None,
    api_calls: int = 0,
    tool_names: Sequence[str] = (),
    response: str | None = None,
) -> TurnTerminalEvent:
    coordinates = _coordinates(event)
    error_type, error_detail = _safe_error(error)
    stable_tools = tuple(dict.fromkeys(str(name) for name in tool_names if name))
    response_sha256 = (
        hashlib.sha256(response.encode("utf-8")).hexdigest() if response else None
    )
    return TurnTerminalEvent(
        schema_version=SCHEMA_VERSION,
        event_id=_stable_id("turn-terminal", event, session_id, run_generation),
        occurred_at=datetime.now(timezone.utc).isoformat(),
        **coordinates,
        session_id=str(session_id or ""),
        run_generation=int(run_generation or 0),
        outcome=str(outcome),
        failure_class=failure_class,
        error_type=error_type,
        error_detail=error_detail,
        api_calls=max(0, int(api_calls or 0)),
        tool_names=stable_tools,
        response_sha256=response_sha256,
    )


def build_delivery_complete_event(*, event: Any, outcome: Any) -> DeliveryCompleteEvent:
    coordinates = _coordinates(event)
    state = getattr(event, "delivery_state", None)
    outcome_value = str(getattr(outcome, "value", outcome))
    attempted = bool(getattr(state, "reply_attempted", False))
    delivered = bool(getattr(state, "reply_delivered", False))
    failed = bool(getattr(state, "reply_failed", False))
    expects_reply = bool(getattr(event, "expects_reply", False))
    if outcome_value == "cancelled":
        failure_class = "cancelled"
    elif expects_reply and not attempted:
        failure_class = "empty_required_reply"
    elif failed or (attempted and not delivered):
        failure_class = "delivery_failure"
    elif outcome_value != "success":
        failure_class = "turn_failure"
    else:
        failure_class = None
    return DeliveryCompleteEvent(
        schema_version=SCHEMA_VERSION,
        event_id=_stable_id("delivery-complete", event),
        occurred_at=datetime.now(timezone.utc).isoformat(),
        **coordinates,
        outcome=outcome_value,
        failure_class=failure_class,
        expects_reply=expects_reply,
        reply_attempted=attempted,
        reply_delivered=delivered,
        reply_failed=failed,
        failure_notice_delivered=bool(
            getattr(state, "failure_notice_delivered", False)
        ),
        reply_message_ids=tuple(
            str(value)
            for value in getattr(state, "reply_message_ids", ())
            if value
        ),
    )


def classify_agent_result(result: Mapping[str, Any]) -> tuple[str, str | None]:
    if result.get("interrupted"):
        return "interrupted", "interrupted"
    if result.get("failed"):
        return "failed", str(result.get("failure_class") or "agent_failure")
    if result.get("partial"):
        return "partial", str(result.get("failure_class") or "partial_result")
    return "succeeded", None


def extract_tool_names(result: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for message in result.get("messages", ()) or ():
        if not isinstance(message, Mapping):
            continue
        for call in message.get("tool_calls", ()) or ():
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            name = function.get("name") if isinstance(function, Mapping) else None
            if name and str(name) not in names:
                names.append(str(name))
    return tuple(names)


def emit_plugin_event(hook_name: str, event: object) -> None:
    """Emit one typed event through the isolated plugin hook surface."""
    from hermes_cli.plugins import invoke_hook

    invoke_hook(hook_name, event=event)


__all__ = [
    "DeliveryCompleteEvent",
    "SCHEMA_VERSION",
    "TurnTerminalEvent",
    "build_delivery_complete_event",
    "build_turn_terminal_event",
    "classify_agent_result",
    "emit_plugin_event",
    "extract_tool_names",
]
