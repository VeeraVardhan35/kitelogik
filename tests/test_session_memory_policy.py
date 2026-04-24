# SPDX-License-Identifier: Apache-2.0
"""
Tests for the pluggable memory-write policy in :class:`AgentSession`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from kitelogik.agents import AgentSession, default_memory_write_policy
from kitelogik.agents.llm import LLMResponse, ToolCall
from kitelogik.memory.models import TrustTier
from kitelogik.tether.models import SessionContext
from tests.conftest import make_mock_llm


def test_default_policy_primary_session():
    ctx = SessionContext(session_id="p", user_role="t", session_scopes=["read"])
    assert default_memory_write_policy(ctx, "k", "v") == TrustTier.EXTERNAL


def test_default_policy_worker_agent():
    ctx = SessionContext(
        session_id="w",
        user_role="t",
        session_scopes=["read"],
        delegation_depth=1,
    )
    assert default_memory_write_policy(ctx, "k", "v") == TrustTier.DELEGATED


async def test_custom_policy_applied_to_write_memory(agent_gate):
    """A policy that downgrades every write to UNTRUSTED should land on the store."""
    ctx = SessionContext(session_id="m", user_role="t", session_scopes=["write_memory"])

    memory = MagicMock()
    captured: dict = {}

    async def _write(*, key, value, trust_tier, source, session_id):
        captured["tier"] = trust_tier
        entry = MagicMock()
        entry.key = key
        entry.trust_tier = trust_tier
        entry.sanitized = False
        return entry

    memory.write = _write

    llm = make_mock_llm(
        responses=[
            LLMResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="tc1", name="write_memory", input={"key": "k", "value": "v"})
                ],
                raw_content="x",
            ),
            LLMResponse(stop_reason="end_turn", text_content="done"),
        ]
    )

    def untrusted_policy(_ctx, _k, _v):
        return TrustTier.UNTRUSTED

    session = AgentSession(
        gate=agent_gate,
        context=ctx,
        llm_client=llm,
        memory_store=memory,
        memory_write_policy=untrusted_policy,
    )
    await session.run_async("store it")
    assert captured["tier"] == TrustTier.UNTRUSTED
