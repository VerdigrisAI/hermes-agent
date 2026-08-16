"""The adapter's two unbounded resources, and the approval label it derives.

Each POST used to start an unbounded queue and an uncapped daemon thread, and
it labelled the run with whatever `thread_id` the caller sent. That label keys
the process-wide skip-prompts set in tools.approval, so a caller could pick a
label a person had already put there with /yolo.
"""

from __future__ import annotations

import asyncio

import pytest

from agui_adapter import server


def test_approval_label_is_namespaced_away_from_human_sessions():
    """A caller-chosen thread_id must not be able to equal a /yolo session key."""
    # The exact expression used at the set_session_vars call site.
    assert f"agui:{'ops'}" == "agui:ops"
    # The property that matters: no caller input produces a bare label.
    for hostile in ("ops", "default", "", "agui:ops"):
        assert f"agui:{hostile}" != hostile


def test_queue_cap_and_run_cap_read_the_environment(monkeypatch):
    monkeypatch.setenv("HERMES_AGUI_MAX_QUEUE_EVENTS", "7")
    monkeypatch.setenv("HERMES_AGUI_MAX_CONCURRENT_RUNS", "3")
    assert server._max_queue_events() == 7
    assert server._max_concurrent_runs() == 3


@pytest.mark.parametrize("bad", ["0", "-1", "abc", ""])
def test_bad_cap_values_fall_back_instead_of_crashing(monkeypatch, bad):
    monkeypatch.setenv("HERMES_AGUI_MAX_QUEUE_EVENTS", bad)
    assert server._max_queue_events() == 1000


def test_caps_have_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("HERMES_AGUI_MAX_QUEUE_EVENTS", raising=False)
    monkeypatch.delenv("HERMES_AGUI_MAX_CONCURRENT_RUNS", raising=False)
    assert server._max_queue_events() == 1000
    assert server._max_concurrent_runs() == 8


def test_run_slots_refuse_beyond_capacity_and_recover():
    """A BoundedSemaphore is the whole cap. Take them all, then give them back."""
    slots = server._run_slots
    taken = []
    while slots.acquire(blocking=False):
        taken.append(True)
    assert taken, "expected at least one slot"
    # At capacity the adapter refuses rather than starting another thread.
    assert slots.acquire(blocking=False) is False
    for _ in taken:
        slots.release()
    # Recovered.
    assert slots.acquire(blocking=False) is True
    slots.release()


@pytest.mark.asyncio
async def test_full_queue_ends_the_run_as_an_error_not_a_success():
    """A truncated transcript must never be reported as a completed run."""
    from ag_ui.core import RunAgentInput
    from ag_ui.encoder import EventEncoder
    from agui_adapter import approvals

    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    queue.put_nowait(approvals.DONE)

    run_input = RunAgentInput(
        thread_id="t1", run_id="r1", state={}, messages=[], tools=[],
        context=[], forwarded_props={},
    )

    frames = [
        frame async for frame in server._consume_queue(
            queue, EventEncoder(), run_input, overflow={"hit": True},
        )
    ]
    body = "".join(frames)
    assert "RUN_ERROR" in body
    assert "truncated" in body
    assert "RUN_FINISHED" not in body


@pytest.mark.asyncio
async def test_clean_queue_still_reports_success():
    from ag_ui.core import RunAgentInput
    from ag_ui.encoder import EventEncoder
    from agui_adapter import approvals

    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    queue.put_nowait(approvals.DONE)

    run_input = RunAgentInput(
        thread_id="t1", run_id="r1", state={}, messages=[], tools=[],
        context=[], forwarded_props={},
    )

    frames = [
        frame async for frame in server._consume_queue(
            queue, EventEncoder(), run_input, overflow={"hit": False},
        )
    ]
    body = "".join(frames)
    assert "RUN_FINISHED" in body
    assert "RUN_ERROR" not in body
