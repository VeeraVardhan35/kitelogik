# SPDX-License-Identifier: Apache-2.0
"""
Tests for AgentSession default behaviours and wiring:

- Default model derived from the LLMClient
- ``run_sync`` wrapper
- ``system_prompt`` override
- Error taxonomy (``LLMProviderError``, ``ToolHandlerError``)
- Single-use guard (``SessionAlreadyRanError``)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kitelogik.agents import (
    DEFAULT_SYSTEM_PROMPT,
    AgentSession,
    LLMProviderError,
    SessionAlreadyRanError,
    ToolHandlerError,
)
from kitelogik.agents.llm import LLMResponse, ToolCall
from tests.conftest import make_mock_llm

# ── default model ───────────────────────────────────────────────────────────


def test_model_defaults_from_llm_client(agent_gate, agent_ctx):
    llm = make_mock_llm(default_model="gpt-4o")
    session = AgentSession(gate=agent_gate, context=agent_ctx, llm_client=llm)
    assert session.model == "gpt-4o"


def test_explicit_model_overrides_default(agent_gate, agent_ctx):
    llm = make_mock_llm(default_model="gpt-4o")
    session = AgentSession(gate=agent_gate, context=agent_ctx, llm_client=llm, model="gpt-4o-mini")
    assert session.model == "gpt-4o-mini"


def test_missing_model_and_default_raises(agent_gate, agent_ctx):
    llm = MagicMock(
        spec=["create_message", "build_tool_result_messages", "format_assistant_message"]
    )
    # Explicitly remove default_model to simulate a client without one
    del llm.default_model
    with pytest.raises(ValueError, match="default_model"):
        AgentSession(gate=agent_gate, context=agent_ctx, llm_client=llm)


# ── system_prompt override ──────────────────────────────────────────────────


def test_system_prompt_defaults(agent_gate, agent_ctx):
    llm = make_mock_llm()
    session = AgentSession(gate=agent_gate, context=agent_ctx, llm_client=llm)
    assert session.system_prompt == DEFAULT_SYSTEM_PROMPT


def test_system_prompt_override(agent_gate, agent_ctx):
    llm = make_mock_llm()
    session = AgentSession(
        gate=agent_gate,
        context=agent_ctx,
        llm_client=llm,
        system_prompt="You are a test agent.",
    )
    assert session.system_prompt == "You are a test agent."


async def test_system_prompt_reaches_llm(agent_gate, agent_ctx):
    llm = make_mock_llm()
    session = AgentSession(
        gate=agent_gate,
        context=agent_ctx,
        llm_client=llm,
        system_prompt="custom prompt",
    )
    await session.run_async("hi")
    _, kwargs = llm.create_message.call_args
    assert kwargs["system"] == "custom prompt"


# ── run_sync ────────────────────────────────────────────────────────────────


def test_run_sync_delegates_to_run_async(agent_gate, agent_ctx):
    llm = make_mock_llm(text="sync result")
    session = AgentSession(gate=agent_gate, context=agent_ctx, llm_client=llm)
    result = session.run_sync("hello")
    assert result.final_response == "sync result"


# ── single-use ──────────────────────────────────────────────────────────────


async def test_session_second_run_raises(agent_gate, agent_ctx):
    llm = make_mock_llm()
    session = AgentSession(gate=agent_gate, context=agent_ctx, llm_client=llm)
    await session.run_async("first")
    with pytest.raises(SessionAlreadyRanError):
        await session.run_async("second")


# ── error wrapping ──────────────────────────────────────────────────────────


async def test_provider_error_wrapped(agent_gate, agent_ctx):
    llm = make_mock_llm()
    llm.create_message = AsyncMock(side_effect=TimeoutError("upstream timeout"))
    session = AgentSession(gate=agent_gate, context=agent_ctx, llm_client=llm)

    with pytest.raises(LLMProviderError) as exc_info:
        await session.run_async("hi")

    assert isinstance(exc_info.value.original, TimeoutError)
    assert exc_info.value.__cause__ is exc_info.value.original


async def test_tool_handler_error_wrapped(agent_gate, agent_ctx):
    llm = make_mock_llm(
        responses=[
            LLMResponse(
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="tc1", name="mytool", input={})],
                raw_content="x",
            )
        ]
    )

    def bad_handler(name: str, args: dict) -> str:
        raise ValueError("handler blew up")

    session = AgentSession(
        gate=agent_gate, context=agent_ctx, llm_client=llm, tool_handler=bad_handler
    )

    with pytest.raises(ToolHandlerError) as exc_info:
        await session.run_async("trigger")

    assert exc_info.value.tool_name == "mytool"
    assert isinstance(exc_info.value.original, ValueError)
