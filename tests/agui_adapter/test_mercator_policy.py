from __future__ import annotations

import contextvars
import asyncio
import json

import pytest
from types import SimpleNamespace

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi.testclient import TestClient

from agui_adapter import server
from agui_adapter.server import MERCATOR_ACCEPTANCE_POLICY_API


class _Contract:
    policy_api_version = 1
    allow_core_tools = False
    use_hermes_approvals = False
    allow_inherited_approval_state = False
    server_toolsets = ()
    server_tool_names = ()
    policy_store = SimpleNamespace()

    def __init__(self) -> None:
        self.started = []

    def frontend_tool_schemas(self):
        return [
            {
                "name": "acceptance_open_surface",
                "description": "Open a fixed route.",
                "parameters": {
                    "type": "object",
                    "properties": {"route": {"type": "string"}},
                    "required": ["route"],
                },
            }
        ]

    def current_principal(self):
        return SimpleNamespace(run_id="arn_123", candidate_id="sha256:" + "a" * 64)

    def ensure_run(self, principal) -> None:
        self.started.append(principal.run_id)


def _body(**changes):
    body = {
        "threadId": "arn_123",
        "runId": "turn_1",
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": "Help me review."}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    body.update(changes)
    return body


def test_policy_bound_factory_rejects_client_selected_tools() -> None:
    contract = _Contract()
    client = TestClient(server.create_mercator_acceptance_app(contract=contract))
    response = client.post(
        "/",
        json=_body(tools=[{"name": "terminal", "description": "no", "parameters": {}}]),
    )
    assert response.status_code == 400
    assert contract.started == []


def test_policy_bound_factory_rejects_client_declared_state_writer_tools() -> None:
    """forwardedProps is a second tool-declaration channel.

    Rejecting body["tools"] alone was not enough: _run_turn feeds
    forwarded_props to translate.parse_state_writer_props and build_run_agent
    then registers a SERVER-EXECUTED handler for every name declared there, so
    a client could grow the surface past contract.frontend_tool_schemas() on
    the one adapter whose whole purpose is a fixed, frontend-only tool list.
    """
    from agui_adapter import translate

    for key in ("forwardedProps", "forwarded_props"):
        contract = _Contract()
        client = TestClient(server.create_mercator_acceptance_app(contract=contract))
        response = client.post(
            "/",
            json=_body(
                **{
                    key: {
                        translate.STATE_WRITER_PROPS_KEY: [
                            {"name": "write_doc", "stateKey": "document"}
                        ]
                    }
                }
            ),
        )
        assert response.status_code == 400, key
        assert contract.started == [], key


def test_policy_bound_factory_allows_unrelated_forwarded_props() -> None:
    """Only the state-writer channel is refused; ordinary props still pass.

    Assert on ensure_run, not on the status code. A policy-bound stream returns
    200 as soon as it starts streaming, so the status proves only that the
    request cleared the pre-stream gate -- the body of this very response is a
    RUN_ERROR, because no model is reachable in the test environment. Reaching
    ensure_run is what proves the props passed every gate.
    """
    contract = _Contract()
    client = TestClient(server.create_mercator_acceptance_app(contract=contract))
    response = client.post("/", json=_body(forwardedProps={"locale": "en-GB"}))
    assert response.status_code == 200
    assert contract.started == ["arn_123"]


def test_policy_bound_factory_rejects_a_thread_from_another_run() -> None:
    """The cross-run guard: one run's principal must not drive another's agent."""
    contract = _Contract()
    client = TestClient(server.create_mercator_acceptance_app(contract=contract))
    response = client.post("/", json=_body(threadId="arn_someone_else"))
    assert response.status_code == 400
    assert contract.started == []


def test_policy_bound_factory_rejects_a_missing_principal() -> None:
    """Fail closed when the outer ASGI layer did not bind a verified principal."""
    contract = _Contract()
    contract.current_principal = lambda: None
    client = TestClient(server.create_mercator_acceptance_app(contract=contract))
    response = client.post("/", json=_body())
    assert response.status_code == 400
    assert contract.started == []


def test_policy_bound_app_refuses_resume() -> None:
    """Resume must not re-attach a parked, terminal-capable worker here.

    The acceptance surface is chat-only. Falling through to approvals.take()
    would re-attach a worker on a surface whose contract is frontend-only with
    Hermes approvals disabled.
    """
    contract = _Contract()
    client = TestClient(server.create_mercator_acceptance_app(contract=contract))
    body = _body()
    body["resume"] = [
        {"interruptId": "int-1", "status": "resolved", "payload": {"approved": True}}
    ]
    response = client.post("/", json=body)
    assert response.status_code == 200
    assert "RUN_STARTED" in response.text
    assert "Acceptance action resume is not enabled." in response.text


def test_policy_bound_factory_injects_exact_frontend_surface(monkeypatch) -> None:
    contract = _Contract()
    captured = {}

    async def fake_stream(run_input, encoder, config, headers, policy_contract=None):
        captured["names"] = {tool.name for tool in run_input.tools}
        captured["toolsets"] = config.enabled_toolsets
        captured["frontend_only"] = config.frontend_only
        captured["contract"] = policy_contract
        yield "data: {}\n\n"

    monkeypatch.setattr(server, "_event_stream", fake_stream)
    client = TestClient(server.create_mercator_acceptance_app(contract=contract))
    response = client.post("/", json=_body())

    assert response.status_code == 200
    assert captured == {
        "names": {"acceptance_open_surface"},
        "toolsets": [],
        "frontend_only": True,
        "contract": contract,
    }
    assert contract.started == ["arn_123"]


@pytest.mark.parametrize(
    "attr, value, fragment",
    [
        ("policy_api_version", MERCATOR_ACCEPTANCE_POLICY_API + 1, "policy API"),
        # bool subclasses int and True == 1, so a loose gate would admit this.
        ("policy_api_version", True, "policy API"),
        ("policy_api_version", float(MERCATOR_ACCEPTANCE_POLICY_API), "policy API"),
        ("policy_api_version", None, "policy API"),
        ("allow_core_tools", True, "core tools"),
        ("server_toolsets", ("hermes-acp",), "zero server tools"),
        ("server_tool_names", ("terminal",), "zero server tools"),
        ("use_hermes_approvals", True, "approval state"),
        ("allow_inherited_approval_state", True, "inherited approval state"),
    ],
)
def test_policy_bound_factory_fails_closed_on_every_gate(attr, value, fragment) -> None:
    """Each contract gate is a separate fail-closed check; cover them all.

    The previous version of this test set only allow_core_tools, despite its
    name claiming server-tool coverage -- so the server_toolsets and
    server_tool_names branches it was named for, plus the policy-API,
    approval-state and inherited-approval gates, never executed.
    """
    bad = _Contract()
    setattr(bad, attr, value)
    with pytest.raises(ValueError, match=fragment):
        server.create_mercator_acceptance_app(contract=bad)


def test_policy_bound_factory_requires_a_non_empty_frontend_surface() -> None:
    """An empty schema list would advertise no tools while still mounting."""
    for schemas in ([], "not-a-list", None):
        bad = _Contract()
        bad.frontend_tool_schemas = lambda schemas=schemas: schemas
        with pytest.raises(ValueError, match="frontend schemas"):
            server.create_mercator_acceptance_app(contract=bad)


def test_worker_context_and_every_frontend_handoff_reach_policy(monkeypatch) -> None:
    principal_var = contextvars.ContextVar("verified_principal", default=None)
    authorized = []
    principal = SimpleNamespace(run_id="arn_123", candidate_id="sha256:" + "a" * 64)
    contract = _Contract()
    contract.current_principal = principal_var.get
    contract.policy_store = SimpleNamespace(authorize=lambda **kwargs: authorized.append(kwargs))

    def fake_turn(run_input, config, bridge, headers, approval_cb=None, on_agent=None):
        return {
            "result": {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "invoke_1",
                                "function": {
                                    "name": "acceptance_open_surface",
                                    "arguments": json.dumps({"route": "/"}),
                                },
                            }
                        ],
                    }
                ]
            },
            "frontend_names": {"acceptance_open_surface"},
            "state_writer_names": set(),
            "run_state": server.RunState(),
        }

    monkeypatch.setattr(server, "_run_turn", fake_turn)
    run_input = RunAgentInput.model_validate({**_body(), "tools": contract.frontend_tool_schemas()})
    async def collect():
        token = principal_var.set(principal)
        try:
            return [
                frame
                async for frame in server._event_stream(
                    run_input,
                    EventEncoder(),
                    server.AgentConfig(),
                    {},
                    policy_contract=contract,
                )
            ]
        finally:
            principal_var.reset(token)

    frames = asyncio.run(collect())

    assert any("TOOL_CALL_START" in frame for frame in frames)
    assert len(authorized) == 1
    assert authorized[0]["run_id"] == "arn_123"
    assert authorized[0]["action_name"] == "acceptance_open_surface"
    assert authorized[0]["arguments"] == {"route": "/"}
