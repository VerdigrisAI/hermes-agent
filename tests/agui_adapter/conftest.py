"""Shared fixtures for the AG-UI adapter test package."""
import pytest

from agui_adapter import approvals


@pytest.fixture(autouse=True)
def _clean_parked_registry():
    """Clear the process-global parked-approval registry around every test.

    The registry (``approvals._parked``) is keyed by ``thread_id`` and shared
    across the whole package. An entry leaked by a test that fails before its
    in-body ``take()``/``discard()`` would otherwise cascade into a later test
    (``register()`` refuses to overwrite, turning a real failure into a
    spurious one downstream). Living in a package ``conftest`` protects both
    ``test_approvals.py`` and ``test_e2e_aimock.py``.
    """
    _release_parked()
    yield
    _release_parked()


def _release_parked() -> None:
    """Deny every pending decision, then drop the entry.

    Clearing the registry alone removes the bookkeeping but leaves
    ``PendingApproval.decision`` unresolved, so a worker thread parked by a
    failed test stays blocked on a future nobody will ever complete -- it only
    unwinds on the approval timeout. Failing closed ("deny") matches the
    timeout path in approvals.py.
    """
    for parked in list(approvals._parked.values()):
        decision = getattr(getattr(parked, "pending", None), "decision", None)
        if decision is not None and not decision.done():
            decision.set_result("deny")
    approvals._parked.clear()
