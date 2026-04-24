# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the non-Anthropic LLMClient implementations.

These tests mock the underlying provider SDK — no network traffic. The goal
is to exercise the protocol contract: ``create_message`` produces a normalised
``LLMResponse``, ``build_tool_result_messages`` emits the provider-specific
message shape, and ``format_assistant_message`` round-trips tool calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kitelogik.agents.llm import LLMResponse

# ─────────────────────────────────────────────────────────────────────────────
# OpenAIClient
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def openai_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    pytest.importorskip("openai")

    from kitelogik.agents.openai_client import OpenAIClient

    client = OpenAIClient()
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
    )
    return client


async def test_openai_text_response(openai_client):
    openai_client._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="hello world",
                    tool_calls=None,
                )
            )
        ]
    )

    resp = await openai_client.create_message(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system="you are a test",
    )

    assert isinstance(resp, LLMResponse)
    assert resp.stop_reason == "end_turn"
    assert resp.text_content == "hello world"
    assert resp.tool_calls == []


async def test_openai_tool_call_response(openai_client):
    tc = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(
            name="get_weather",
            arguments=json.dumps({"city": "Amsterdam"}),
        ),
    )
    openai_client._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))]
    )

    resp = await openai_client.create_message(
        model="gpt-4o",
        messages=[{"role": "user", "content": "what's the weather?"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
        system="",
    )

    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "call_123"
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].input == {"city": "Amsterdam"}


def test_openai_build_tool_result_messages(openai_client):
    msgs = openai_client.build_tool_result_messages(
        [("call_1", "result one"), ("call_2", "result two")]
    )
    assert msgs == [
        {"role": "tool", "tool_call_id": "call_1", "content": "result one"},
        {"role": "tool", "tool_call_id": "call_2", "content": "result two"},
    ]


def test_openai_format_assistant_message_roundtrip(openai_client):
    raw = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_9",
                function=SimpleNamespace(name="search", arguments='{"q":"x"}'),
            )
        ],
    )
    msg = openai_client.format_assistant_message(raw)
    assert msg["role"] == "assistant"
    assert "content" not in msg  # empty content dropped
    assert msg["tool_calls"][0]["id"] == "call_9"
    assert msg["tool_calls"][0]["function"]["name"] == "search"


def test_openai_handles_malformed_tool_arguments(openai_client, monkeypatch):
    tc = SimpleNamespace(
        id="call_bad",
        function=SimpleNamespace(name="oops", arguments="not-json"),
    )
    openai_client._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))]
        )
    )

    import asyncio

    resp = asyncio.run(
        openai_client.create_message(model="gpt-4o", messages=[], tools=[], system="")
    )
    assert resp.tool_calls[0].input == {"_raw": "not-json"}


# ─────────────────────────────────────────────────────────────────────────────
# GoogleClient
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def google_client(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "ya29.test")

    from kitelogik.agents.google_client import GoogleClient

    client = GoogleClient()
    client._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock()))
    )
    return client


async def test_google_text_response(google_client):
    google_client._client.aio.models.generate_content.return_value = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="hallo", function_call=None)])
            )
        ]
    )

    resp = await google_client.create_message(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system="you are a test",
    )

    assert resp.stop_reason == "end_turn"
    assert resp.text_content == "hallo"
    assert resp.tool_calls == []


async def test_google_tool_call_response(google_client):
    fc = SimpleNamespace(name="get_weather", args={"city": "Utrecht"})
    google_client._client.aio.models.generate_content.return_value = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=None, function_call=fc)])
            )
        ]
    )

    resp = await google_client.create_message(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "get_weather", "parameters": {}}],
        system="",
    )

    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].input == {"city": "Utrecht"}


def test_google_build_tool_result_messages(google_client):
    msgs = google_client.build_tool_result_messages(
        [("get_weather", '{"temp": 12}'), ("search", "raw string result")]
    )
    assert msgs[0]["role"] == "function"
    assert msgs[0]["parts"][0]["function_response"]["response"] == {"temp": 12}
    assert msgs[1]["parts"][0]["function_response"]["response"] == {"result": "raw string result"}
