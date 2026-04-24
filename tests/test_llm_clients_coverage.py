# SPDX-License-Identifier: Apache-2.0
"""
Coverage tests for ``kitelogik/agents/{llm,openai_client,google_client}.py``.

Exercises constructor error paths, default_model overrides, real-SDK
response parsing (via mocked SDK objects), and ``is_retryable_error``
edge cases.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kitelogik.agents.llm import (
    AnthropicLLMClient,
    is_retryable_error,
)

# ─────────────────────────────────────────────────────────────────────────────
# is_retryable_error edge cases
# ─────────────────────────────────────────────────────────────────────────────


class _ExcWith:
    """Helper: build an exception subclass with a given attribute."""

    def __new__(cls, attr: str, value):
        E = type("E", (Exception,), {attr: value})
        return E("boom")


def test_retryable_none_status_is_transient():
    assert is_retryable_error(TimeoutError("slow network")) is True


def test_retryable_status_429():
    assert is_retryable_error(_ExcWith("status_code", 429)) is True


def test_retryable_status_500_series():
    assert is_retryable_error(_ExcWith("status_code", 502)) is True


def test_retryable_status_400_is_not():
    assert is_retryable_error(_ExcWith("status_code", 400)) is False


def test_retryable_status_404_is_not():
    assert is_retryable_error(_ExcWith("status_code", 404)) is False


def test_retryable_falls_back_to_status_attribute():
    assert is_retryable_error(_ExcWith("status", 503)) is True


def test_retryable_non_integer_status_treated_as_transient():
    """The fallback on ``int(status)`` failing: treat as transient."""
    assert is_retryable_error(_ExcWith("status_code", "not-an-int")) is True


def test_retryable_none_object_status_is_transient():
    # getattr returns None for both .status_code and .status → transient
    assert is_retryable_error(Exception("generic")) is True


# ─────────────────────────────────────────────────────────────────────────────
# AnthropicLLMClient — constructor + real-SDK response parsing
# ─────────────────────────────────────────────────────────────────────────────


def test_anthropic_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicLLMClient(api_key=None)


def test_anthropic_default_model_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy")
    client = AnthropicLLMClient(default_model="claude-opus-4-6")
    assert client.default_model == "claude-opus-4-6"


async def test_anthropic_create_message_parses_text_block(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy")
    client = AnthropicLLMClient()

    # Anthropic response shape: .content is a list of blocks; each block has
    # .type, and text blocks additionally have .text.
    text_block = SimpleNamespace(type="text", text="hello world")
    response = SimpleNamespace(
        content=[text_block],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=12, output_tokens=3),
    )
    client._client.messages.create = AsyncMock(return_value=response)

    resp = await client.create_message(model="claude-sonnet-4-6", messages=[], tools=[], system="")
    assert resp.stop_reason == "end_turn"
    assert resp.text_content == "hello world"
    assert resp.input_tokens == 12
    assert resp.output_tokens == 3
    assert resp.tool_calls == []


async def test_anthropic_create_message_parses_tool_use(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy")
    client = AnthropicLLMClient()

    tool_block = SimpleNamespace(
        type="tool_use",
        id="tu_001",
        name="lookup",
        input={"q": "x"},
    )
    response = SimpleNamespace(
        content=[tool_block],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=5, output_tokens=1),
    )
    client._client.messages.create = AsyncMock(return_value=response)

    resp = await client.create_message(model="claude-sonnet-4-6", messages=[], tools=[], system="")
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "tu_001"
    assert resp.tool_calls[0].name == "lookup"
    assert resp.tool_calls[0].input == {"q": "x"}


async def test_anthropic_create_message_without_usage(monkeypatch):
    """Response with no ``usage`` attribute should yield None token counts."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy")
    client = AnthropicLLMClient()

    text_block = SimpleNamespace(type="text", text="hi")
    # Build a minimal object without usage attribute.
    response = SimpleNamespace(content=[text_block], stop_reason="end_turn", usage=None)
    client._client.messages.create = AsyncMock(return_value=response)

    resp = await client.create_message(model="claude-sonnet-4-6", messages=[], tools=[], system="")
    assert resp.input_tokens is None
    assert resp.output_tokens is None


def test_anthropic_build_tool_result_messages_wraps_in_user():
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)  # avoid __init__
    msgs = client.build_tool_result_messages([("t1", "out1"), ("t2", "out2")])
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert len(msgs[0]["content"]) == 2
    assert msgs[0]["content"][0]["tool_use_id"] == "t1"


def test_anthropic_format_assistant_message_roundtrips():
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    raw = [SimpleNamespace(type="text", text="hi")]
    msg = client.format_assistant_message(raw)
    assert msg == {"role": "assistant", "content": raw}


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIClient — constructor error paths
# ─────────────────────────────────────────────────────────────────────────────


def test_openai_missing_api_key_raises(monkeypatch):
    pytest.importorskip("openai")
    from kitelogik.agents.openai_client import OpenAIClient

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        OpenAIClient(api_key=None)


def test_openai_default_model_override(monkeypatch):
    pytest.importorskip("openai")
    from kitelogik.agents.openai_client import OpenAIClient

    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    client = OpenAIClient(default_model="gpt-4o-mini")
    assert client.default_model == "gpt-4o-mini"


def test_openai_format_assistant_message_text_only(monkeypatch):
    pytest.importorskip("openai")
    from kitelogik.agents.openai_client import OpenAIClient

    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    client = OpenAIClient()

    # Message object with only content, no tool_calls
    raw = SimpleNamespace(content="just a text answer", tool_calls=None)
    msg = client.format_assistant_message(raw)
    assert msg == {"role": "assistant", "content": "just a text answer"}


# ─────────────────────────────────────────────────────────────────────────────
# GoogleClient — constructor error paths + helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_google_missing_api_key_raises(monkeypatch):
    pytest.importorskip("google.genai")
    from kitelogik.agents.google_client import GoogleClient

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        GoogleClient(api_key=None)


def test_google_default_model_override(monkeypatch):
    pytest.importorskip("google.genai")
    from kitelogik.agents.google_client import GoogleClient

    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    client = GoogleClient(default_model="gemini-2.5-pro")
    assert client.default_model == "gemini-2.5-pro"


def test_google_parse_maybe_json_dict():
    pytest.importorskip("google.genai")
    from kitelogik.agents.google_client import GoogleClient

    assert GoogleClient._parse_maybe_json('{"a": 1}') == {"a": 1}


def test_google_parse_maybe_json_non_dict_wrapped():
    """Non-dict JSON (e.g. an int) is wrapped as ``{"result": ...}``."""
    pytest.importorskip("google.genai")
    from kitelogik.agents.google_client import GoogleClient

    assert GoogleClient._parse_maybe_json("42") == {"result": 42}


def test_google_parse_maybe_json_invalid_json_wrapped():
    pytest.importorskip("google.genai")
    from kitelogik.agents.google_client import GoogleClient

    assert GoogleClient._parse_maybe_json("not-json") == {"result": "not-json"}


def test_google_messages_to_contents_passthrough_if_gemini_shape():
    pytest.importorskip("google.genai")
    from kitelogik.agents.google_client import GoogleClient

    already = [{"role": "user", "parts": [{"text": "hi"}]}]
    assert GoogleClient._messages_to_contents(already) == already


def test_google_messages_to_contents_wraps_plain_text():
    pytest.importorskip("google.genai")
    from kitelogik.agents.google_client import GoogleClient

    plain = [{"role": "assistant", "content": "hello"}]
    out = GoogleClient._messages_to_contents(plain)
    # "assistant" maps to "model" for Gemini
    assert out == [{"role": "model", "parts": [{"text": "hello"}]}]


def test_google_messages_to_contents_passes_list_content():
    pytest.importorskip("google.genai")
    from kitelogik.agents.google_client import GoogleClient

    plain = [{"role": "user", "content": [{"text": "chunk1"}, {"text": "chunk2"}]}]
    out = GoogleClient._messages_to_contents(plain)
    assert out == [{"role": "user", "parts": [{"text": "chunk1"}, {"text": "chunk2"}]}]


def test_google_format_assistant_message_with_text_and_tool_call(monkeypatch):
    pytest.importorskip("google.genai")
    from kitelogik.agents.google_client import GoogleClient

    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    client = GoogleClient()

    # Mirror the Gemini response shape: candidates[].content.parts[].{text,function_call}
    fc = SimpleNamespace(name="lookup", args={"q": "x"})
    parts = [
        SimpleNamespace(text="narration", function_call=None),
        SimpleNamespace(text=None, function_call=fc),
    ]
    candidate = SimpleNamespace(content=SimpleNamespace(parts=parts))
    raw = SimpleNamespace(candidates=[candidate])

    msg = client.format_assistant_message(raw)
    assert msg["role"] == "model"
    assert msg["parts"] == [
        {"text": "narration"},
        {"function_call": {"name": "lookup", "args": {"q": "x"}}},
    ]


def test_google_format_assistant_message_empty_candidates(monkeypatch):
    pytest.importorskip("google.genai")
    from kitelogik.agents.google_client import GoogleClient

    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    client = GoogleClient()

    raw = SimpleNamespace(candidates=[])
    msg = client.format_assistant_message(raw)
    assert msg == {"role": "model", "parts": []}
