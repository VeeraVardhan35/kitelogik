# SPDX-License-Identifier: Apache-2.0
"""
Tests for :class:`AgentSession` retry and provider-fallback behaviour.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kitelogik.agents import AgentSession, LLMProviderError
from kitelogik.agents.llm import LLMResponse, RetryConfig
from tests.conftest import make_mock_llm


class _RateLimit(Exception):
    status_code = 429


class _BadRequest(Exception):
    status_code = 400


class _ServerErr(Exception):
    status_code = 503


def _llm_with_side_effect(side_effect) -> MagicMock:
    """Build a mock LLM whose ``create_message`` drives from the given side effect."""
    m = make_mock_llm()
    m.create_message = AsyncMock(side_effect=side_effect)
    return m


FAST_RETRY = RetryConfig(max_retries=2, initial_delay=0.01, backoff_factor=1.5)


async def test_retry_on_429_then_success(agent_gate, agent_ctx):
    good = LLMResponse(stop_reason="end_turn", text_content="ok")
    llm = _llm_with_side_effect([_RateLimit("slow down"), good])
    session = AgentSession(
        gate=agent_gate, context=agent_ctx, llm_client=llm, retry_config=FAST_RETRY
    )

    result = await session.run_async("hi")
    assert result.final_response == "ok"
    assert llm.create_message.call_count == 2


async def test_retry_exhausted_raises(agent_gate, agent_ctx):
    llm = _llm_with_side_effect([_RateLimit("1"), _RateLimit("2"), _RateLimit("3")])
    session = AgentSession(
        gate=agent_gate, context=agent_ctx, llm_client=llm, retry_config=FAST_RETRY
    )

    with pytest.raises(LLMProviderError):
        await session.run_async("hi")
    assert llm.create_message.call_count == 3  # 1 initial + 2 retries


async def test_4xx_not_retried(agent_gate, agent_ctx):
    llm = _llm_with_side_effect([_BadRequest("nope")])
    session = AgentSession(
        gate=agent_gate,
        context=agent_ctx,
        llm_client=llm,
        retry_config=RetryConfig(max_retries=5, initial_delay=0.01, backoff_factor=1.5),
    )

    with pytest.raises(LLMProviderError):
        await session.run_async("hi")
    assert llm.create_message.call_count == 1


async def test_5xx_retried_and_fallback_used(agent_gate, agent_ctx):
    primary = _llm_with_side_effect([_ServerErr("oops"), _ServerErr("oops")])
    fallback = _llm_with_side_effect(
        [LLMResponse(stop_reason="end_turn", text_content="from fallback")]
    )
    session = AgentSession(
        gate=agent_gate,
        context=agent_ctx,
        llm_client=primary,
        fallback_llm_client=fallback,
        retry_config=RetryConfig(max_retries=1, initial_delay=0.01, backoff_factor=1.5),
    )

    events: list[dict] = []
    result = await session.run_async("hi", on_event=events.append)
    assert result.final_response == "from fallback"
    assert any(e["type"] == "llm_fallback" for e in events)
    assert any(e["type"] == "llm_retry" for e in events)


async def test_fallback_also_fails(agent_gate, agent_ctx):
    primary = _llm_with_side_effect([_ServerErr("a")])
    fallback = _llm_with_side_effect([_BadRequest("b")])
    session = AgentSession(
        gate=agent_gate,
        context=agent_ctx,
        llm_client=primary,
        fallback_llm_client=fallback,
        retry_config=RetryConfig(max_retries=0, initial_delay=0.01, backoff_factor=1.5),
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await session.run_async("hi")
    # The last error comes from the fallback
    assert isinstance(exc_info.value.original, _BadRequest)
