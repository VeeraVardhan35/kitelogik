# SPDX-License-Identifier: Apache-2.0
"""
Tests for per-iteration ``agent.budget`` governance in :class:`AgentSession`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kitelogik.agents import AgentSession
from kitelogik.agents.llm import LLMResponse
from kitelogik.tether.gate import PolicyGate
from kitelogik.tether.models import SessionContext
from tests.conftest import allow_decision_for, deny_decision_for


@pytest.fixture
def llm_with_usage() -> MagicMock:
    """Mock LLM returning a fixed token usage so tests can assert counters."""
    m = MagicMock()
    m.default_model = "claude-sonnet-4-6"
    m.create_message = AsyncMock(
        return_value=LLMResponse(
            stop_reason="end_turn",
            text_content="done",
            input_tokens=50,
            output_tokens=10,
        )
    )
    m.build_tool_result_messages = MagicMock(return_value=[])
    m.format_assistant_message = MagicMock(return_value={"role": "assistant", "content": ""})
    return m


def _gate_with_decisions(*decisions) -> PolicyGate:
    """Build a gate mock whose ``evaluate`` returns the given decisions in order."""
    g = MagicMock(spec=PolicyGate)
    g.evaluate = AsyncMock(side_effect=list(decisions))
    g.evaluate_tool_call = AsyncMock(return_value=allow_decision_for())
    g.sanitize_response = MagicMock(
        side_effect=lambda x: MagicMock(content=x, was_modified=False, injection_patterns_found=[])
    )
    return g


async def test_budget_check_skipped_when_no_budget_set(llm_with_usage):
    ctx = SessionContext(session_id="nb", user_role="t", session_scopes=["read"])
    gate = _gate_with_decisions(allow_decision_for())

    session = AgentSession(gate=gate, context=ctx, llm_client=llm_with_usage)
    await session.run_async("go")

    # Only agent.spawn should hit gate.evaluate — no agent.budget
    types = [c.args[0].event_type for c in gate.evaluate.call_args_list]
    assert types == ["agent.spawn"]


async def test_budget_counters_update_when_budget_set(llm_with_usage):
    ctx = SessionContext(
        session_id="b1",
        user_role="t",
        session_scopes=["read"],
        budget_total_tokens=1000,
        budget_used_tokens=0,
    )
    gate = _gate_with_decisions(allow_decision_for(), allow_decision_for())

    session = AgentSession(gate=gate, context=ctx, llm_client=llm_with_usage)
    await session.run_async("go")

    # After a single LLM turn, used_tokens should equal 50 + 10
    assert session.context.budget_used_tokens == 60
    types = [c.args[0].event_type for c in gate.evaluate.call_args_list]
    assert "agent.budget" in types


async def test_budget_deny_halts_session(llm_with_usage):
    ctx = SessionContext(
        session_id="b2",
        user_role="t",
        session_scopes=["read"],
        budget_total_tokens=10,
        budget_used_tokens=0,
    )
    gate = _gate_with_decisions(allow_decision_for(), deny_decision_for(reason="token cap hit"))

    session = AgentSession(gate=gate, context=ctx, llm_client=llm_with_usage)
    events: list[dict] = []
    result = await session.run_async("go", on_event=events.append)

    assert any(e["type"] == "budget_exhausted" for e in events)
    assert result.final_response in {"done", "Session halted: token cap hit"}
