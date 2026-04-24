# SPDX-License-Identifier: Apache-2.0
"""
LLM client abstraction for AgentSession.

Defines a Protocol that any LLM provider can implement, plus the default
AnthropicLLMClient that wraps the Anthropic Python SDK.

To use a non-Anthropic provider with AgentSession, implement the
LLMClient protocol and pass it via ``llm_client=`` at construction::

    session = AgentSession(
        gate=gate,
        context=context,
        llm_client=MyOpenAIClient(api_key="..."),
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Default upper bound on tokens per LLM response. Anthropic's messages API
# requires an explicit ``max_tokens`` value; 4096 is large enough for most
# agent turns (including multi-tool-use sequences) without spiking cost on
# runaway generations. Callers override via the ``max_tokens`` argument.
DEFAULT_MAX_TOKENS = 4096


@dataclass
class RetryConfig:
    """Retry and fallback configuration for LLM provider calls.

    Attributes
    ----------
    max_retries : int
        Number of retry attempts *after* the first call, so total attempts =
        ``1 + max_retries``. Default ``2``.
    initial_delay : float
        Seconds to wait before the first retry. Default ``0.5``.
    backoff_factor : float
        Multiplier applied to the delay between retries. Default ``2.0`` —
        doubles each attempt.
    max_delay : float
        Upper bound on any single backoff interval, in seconds. Default
        ``10.0``.

    Notes
    -----
    The delay applied to the *n*-th retry (0-indexed) is::

        min(initial_delay * backoff_factor ** n + uniform(0, 0.1), max_delay)

    The small uniform jitter prevents thundering-herd retry storms when many
    sessions fail simultaneously.
    """

    max_retries: int = 2
    initial_delay: float = 0.5
    backoff_factor: float = 2.0
    max_delay: float = 10.0


def is_retryable_error(exc: Exception) -> bool:
    """Classify an LLM provider exception as transient vs. fatal.

    Parameters
    ----------
    exc : Exception
        The exception raised by a provider SDK call.

    Returns
    -------
    bool
        ``True`` if the caller should retry (transient), ``False`` if the
        error is a client-side bug that would recur (fatal).

    Notes
    -----
    Classification is status-code-based:

    - Status ``429`` (rate limit) or ``>= 500`` → retry.
    - Status ``4xx`` other than ``429`` → do not retry.
    - No ``status_code`` / ``status`` attribute → assume transient
      (covers ``TimeoutError``, ``ConnectionError``, etc.).
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is None:
        return True
    try:
        status = int(status)
    except (TypeError, ValueError):
        return True
    if status == 429:
        return True
    return status >= 500


@dataclass
class ToolCall:
    """A tool call extracted from an LLM response.

    Attributes
    ----------
    id : str
        Provider-assigned tool-call identifier (used to correlate tool results).
    name : str
        Name of the tool the model requested.
    input : dict[str, Any]
        Parsed arguments the model passed to the tool.
    """

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Provider-neutral response from a single LLM invocation.

    Attributes
    ----------
    stop_reason : str
        Either ``"end_turn"`` (model produced a final answer) or
        ``"tool_use"`` (model requested one or more tool calls).
    text_content : str or None
        The model's text output, if any. ``None`` when the model returned
        only tool calls.
    tool_calls : list[ToolCall]
        Tool calls extracted from the response. Empty when ``stop_reason`` is
        ``"end_turn"``.
    raw_content : Any
        Provider-specific content block kept so it can be round-tripped back
        into conversation history via ``format_assistant_message``.
    input_tokens : int or None
        Prompt tokens consumed, if the provider reports them. Populated by all
        shipped clients when available.
    output_tokens : int or None
        Completion tokens generated, if the provider reports them.
    """

    stop_reason: str  # "end_turn" | "tool_use"
    text_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: Any = None  # Provider-specific content for message history
    input_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Provider protocol the :class:`AgentSession` talks to.

    Any class implementing this three-method protocol can be passed as
    ``llm_client=`` to :class:`AgentSession`. Kite Logik ships
    :class:`AnthropicLLMClient`, :class:`~kitelogik.agents.openai_client.OpenAIClient`,
    and :class:`~kitelogik.agents.google_client.GoogleClient`; custom
    implementations cover Bedrock, Azure OpenAI, local models, etc.

    Attributes
    ----------
    default_model : str
        Model identifier used when the caller constructs ``AgentSession``
        without passing ``model=`` explicitly.

    Notes
    -----
    Streaming is optional. Implementations **may** provide a ``stream_message``
    coroutine with the same signature as ``create_message`` plus an
    ``on_chunk: Callable[[str], None]`` kwarg; :class:`AgentSession` falls
    back to ``create_message`` when ``stream_message`` is absent.
    """

    default_model: str

    async def create_message(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict],
        system: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResponse:
        """Send one request to the provider and return a normalised response.

        Parameters
        ----------
        model : str
            Provider-specific model identifier.
        messages : list[dict]
            Conversation history in the provider's native message shape.
        tools : list[dict]
            Tool schemas in the provider's native format.
        system : str
            System prompt.
        max_tokens : int, optional
            Upper bound on generated tokens. Defaults to
            :data:`DEFAULT_MAX_TOKENS`.

        Returns
        -------
        LLMResponse
            Normalised response with stop reason, text, tool calls, and
            token usage when available.
        """
        ...

    def build_tool_result_messages(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Translate tool results into provider-specific history messages.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            ``(tool_call_id, tool_output)`` pairs in the order the model
            requested them.

        Returns
        -------
        list[dict]
            Messages to ``extend`` onto the conversation history. Anthropic
            wraps all results in one ``role="user"`` message; OpenAI emits
            one ``role="tool"`` message per result; Gemini uses
            ``role="function"`` entries. Returning a list is the single
            join point for that shape difference.
        """
        ...

    def format_assistant_message(self, raw_content: Any) -> dict:
        """Build the assistant-turn message to append to history.

        Parameters
        ----------
        raw_content : Any
            The ``raw_content`` field of a prior :class:`LLMResponse`.

        Returns
        -------
        dict
            A single message in the provider's native shape, ready to
            append to ``messages`` before calling ``create_message`` again.
        """
        ...


class AnthropicLLMClient:
    """:class:`LLMClient` implementation using the Anthropic Python SDK.

    This is the default client — used by :class:`AgentSession` when the
    caller does not pass ``llm_client=``.

    Parameters
    ----------
    api_key : str or None, optional
        Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY``
        environment variable when not provided.
    default_model : str or None, optional
        Model to use when :class:`AgentSession` is created without an
        explicit ``model=``. Defaults to ``"claude-sonnet-4-6"``.

    Raises
    ------
    RuntimeError
        If no API key is provided and ``ANTHROPIC_API_KEY`` is not set.

    Examples
    --------
    >>> from kitelogik.agents.llm import AnthropicLLMClient
    >>> client = AnthropicLLMClient()  # reads ANTHROPIC_API_KEY
    >>> session = AgentSession(gate=gate, context=ctx, llm_client=client)
    """

    default_model: str = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
    ) -> None:
        import anthropic

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or set the environment variable before running."
            )
        self._client = anthropic.AsyncAnthropic(api_key=key)
        if default_model:
            self.default_model = default_model

    async def create_message(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict],
        system: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResponse:
        """Send a message to Claude and return a normalised response.

        Parameters
        ----------
        model : str
            Claude model identifier (e.g. ``"claude-sonnet-4-6"``).
        messages : list[dict]
            Conversation history in Anthropic's messages format.
        tools : list[dict]
            Tool schemas in Anthropic's format
            (``{"name": ..., "description": ..., "input_schema": ...}``).
        system : str
            System prompt.
        max_tokens : int, optional
            Upper bound on generated tokens (required by the Anthropic API).

        Returns
        -------
        LLMResponse
            Normalised response with stop reason, text, tool calls, and
            token usage.
        """
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=tools,  # type: ignore[arg-type]
            system=system,
            messages=messages,  # type: ignore[arg-type]
        )

        text_content = None
        tool_calls = []

        for block in response.content:
            if hasattr(block, "text"):
                text_content = block.text
            if block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input),
                    )
                )

        usage = getattr(response, "usage", None)
        return LLMResponse(
            stop_reason=response.stop_reason,  # type: ignore[arg-type]
            text_content=text_content,
            tool_calls=tool_calls,
            raw_content=response.content,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
        )

    def build_tool_result_messages(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Wrap all tool results in a single Anthropic user message.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            ``(tool_call_id, output)`` pairs.

        Returns
        -------
        list[dict]
            A single-element list with one ``role="user"`` message whose
            ``content`` is a list of ``tool_result`` blocks — the shape
            Anthropic's messages API expects.
        """
        content = [
            {"type": "tool_result", "tool_use_id": tc_id, "content": out} for tc_id, out in pairs
        ]
        return [{"role": "user", "content": content}]

    def format_assistant_message(self, raw_content: Any) -> dict:
        """Wrap Anthropic raw content as an ``assistant`` message.

        Parameters
        ----------
        raw_content : Any
            Content blocks from a prior :class:`LLMResponse.raw_content`.

        Returns
        -------
        dict
            Message ready to append to ``messages``.
        """
        return {"role": "assistant", "content": raw_content}
