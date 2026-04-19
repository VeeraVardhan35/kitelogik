# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI Agents SDK adapter.

openai-agents is not a hard dependency. The governance-flow tests install a
minimal ``agents`` module stub via ``sys.modules`` with a ``FunctionTool``
class that retains the ``on_invoke_tool`` coroutine, which is enough to
drive the policy gate and assert allow/deny behavior."""

import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from kitelogik.tether.gate import PolicyGate
from kitelogik.tether.models import PolicyDecision, RiskTier, SessionContext


@pytest.fixture
def ctx():
    return SessionContext(session_id="test", user_role="admin", session_scopes=["read"])


@pytest.fixture
def mock_gate():
    gate = MagicMock(spec=PolicyGate)
    gate.evaluate_tool_call = AsyncMock(
        return_value=PolicyDecision(
            allow=True,
            deny=False,
            risk_tier=RiskTier.INFORMATIONAL,
            requires_hitl=False,
            reason="Allowed",
        )
    )
    gate.sanitize_response = MagicMock(return_value=MagicMock(content="result", was_modified=False))
    return gate


@pytest.fixture
def deny_gate():
    gate = MagicMock(spec=PolicyGate)
    gate.evaluate_tool_call = AsyncMock(
        return_value=PolicyDecision(
            allow=False,
            deny=True,
            risk_tier=RiskTier.SECURITY_CRITICAL,
            requires_hitl=False,
            reason="Denied by policy",
        )
    )
    return gate


@pytest.fixture
def stub_openai_agents(monkeypatch):
    """Install a minimal ``agents.FunctionTool`` stub so agent_tools() works
    without the real openai-agents package. FunctionTool just stores its
    kwargs so tests can pull out ``on_invoke_tool`` and await it."""
    agents_pkg = types.ModuleType("agents")

    class FunctionTool:  # noqa: D401 — stub mirrors the real class surface used here
        def __init__(self, name, description, params_json_schema, on_invoke_tool):
            self.name = name
            self.description = description
            self.params_json_schema = params_json_schema
            self.on_invoke_tool = on_invoke_tool

    agents_pkg.FunctionTool = FunctionTool
    monkeypatch.setitem(sys.modules, "agents", agents_pkg)
    yield FunctionTool


def test_openai_agents_import_guard():
    """Import guard raises clear error when openai-agents is not installed."""
    from kitelogik.adapters.openai_agents import _require_openai_agents

    try:
        _require_openai_agents()
    except ImportError as e:
        assert "openai-agents" in str(e).lower()


def test_openai_agents_adapter_register(mock_gate, ctx):
    from kitelogik.adapters.openai_agents import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter(gate=mock_gate, context=ctx)
    result = adapter.register("search", lambda q: f"found: {q}", description="Search")
    assert result is adapter  # chaining


def test_openai_agents_adapter_register_with_params(mock_gate, ctx):
    from kitelogik.adapters.openai_agents import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter(gate=mock_gate, context=ctx)
    adapter.register(
        "search",
        lambda query: f"found: {query}",
        description="Search",
        params={"query": {"type": "string"}},
    )
    assert "search" in adapter._tools
    assert adapter._tools["search"][3] == {"query": {"type": "string"}}


# ── Governance-flow tests (use the stub_openai_agents fixture) ─────────────


def test_agent_tools_builds_function_tool_with_schema(stub_openai_agents, mock_gate, ctx):
    from kitelogik.adapters.openai_agents import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter(gate=mock_gate, context=ctx)
    adapter.register(
        "search",
        lambda query: f"found:{query}",
        description="Search docs",
        params={"query": {"type": "string"}},
    )

    [tool] = adapter.agent_tools()
    assert tool.name == "search"
    assert tool.description == "Search docs"
    assert tool.params_json_schema == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }


async def test_agent_tool_allowed_call_runs_and_sanitizes(stub_openai_agents, mock_gate, ctx):
    from kitelogik.adapters.openai_agents import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter(gate=mock_gate, context=ctx)
    adapter.register("search", lambda query: f"found:{query}")

    [tool] = adapter.agent_tools()
    out = await tool.on_invoke_tool(query="widgets")

    assert out == "result"  # sanitize_response fixture
    mock_gate.evaluate_tool_call.assert_awaited_once()
    mock_gate.sanitize_response.assert_called_once()


async def test_agent_tool_denied_call_returns_blocked_json_and_skips_fn(
    stub_openai_agents, deny_gate, ctx
):
    from kitelogik.adapters.openai_agents import OpenAIAgentsAdapter

    called = {"n": 0}

    def should_never_run(**_kw):
        called["n"] += 1
        return "should_not_happen"

    adapter = OpenAIAgentsAdapter(gate=deny_gate, context=ctx)
    adapter.register("delete_all", should_never_run)

    [tool] = adapter.agent_tools()
    out = await tool.on_invoke_tool()

    payload = json.loads(out)
    assert payload["blocked"] is True
    assert "Denied by policy" in payload["reason"]
    assert called["n"] == 0


async def test_agent_tool_async_fn_is_awaited(stub_openai_agents, mock_gate, ctx):
    from kitelogik.adapters.openai_agents import OpenAIAgentsAdapter

    async def async_fn(x):
        return f"async:{x}"

    adapter = OpenAIAgentsAdapter(gate=mock_gate, context=ctx)
    adapter.register("atool", async_fn)

    [tool] = adapter.agent_tools()
    out = await tool.on_invoke_tool(x="hello")
    assert out == "result"
