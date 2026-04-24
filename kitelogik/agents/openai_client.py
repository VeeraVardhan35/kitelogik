# SPDX-License-Identifier: Apache-2.0
"""
OpenAI implementation of the LLMClient protocol.

Install the optional extra::

    pip install 'kitelogik[openai]'

Then pass an ``OpenAIClient`` to ``AgentSession``::

    from kitelogik.agents.openai_client import OpenAIClient

    session = AgentSession(gate=gate, context=ctx, llm_client=OpenAIClient())

Tool schemas passed via ``tools=`` must follow OpenAI's function-calling format
(``{"type": "function", "function": {"name": ..., "parameters": ...}}``), not
Anthropic's.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from kitelogik.agents.llm import DEFAULT_MAX_TOKENS, LLMResponse, ToolCall

logger = logging.getLogger(__name__)


class OpenAIClient:
    """:class:`LLMClient` implementation for OpenAI Chat Completions (tools API).

    Parameters
    ----------
    api_key : str or None, optional
        OpenAI API key. Falls back to ``OPENAI_API_KEY`` when not provided.
    default_model : str or None, optional
        Model used when :class:`AgentSession` is constructed without an
        explicit ``model=``. Defaults to ``"gpt-4o"``.
    base_url : str or None, optional
        Override the API base URL. Useful for Azure OpenAI, OpenRouter,
        or local OpenAI-compatible servers (Ollama, vLLM).

    Raises
    ------
    RuntimeError
        If the ``openai`` package is not installed, or if no API key is
        provided and ``OPENAI_API_KEY`` is not set.

    Examples
    --------
    >>> from kitelogik.agents.openai_client import OpenAIClient
    >>> client = OpenAIClient(default_model="gpt-4o-mini")
    >>> session = AgentSession(gate=gate, context=ctx, llm_client=client)
    """

    default_model: str = "gpt-4o"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            import openai
        except ImportError as e:
            raise RuntimeError(
                "openai is not installed. Install the optional extra: "
                "`pip install 'kitelogik[openai]'` or `pip install 'openai>=1.0'`."
            ) from e

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Pass api_key=... or set the environment variable."
            )
        self._client = openai.AsyncOpenAI(api_key=key, base_url=base_url)
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
        """Send a Chat Completions request and return a normalised response.

        Parameters
        ----------
        model : str
            OpenAI model identifier.
        messages : list[dict]
            Conversation history. Must NOT include a system message; the
            ``system`` argument is prepended automatically.
        tools : list[dict]
            Tools in OpenAI's function-calling format
            (``{"type": "function", "function": {...}}``).
        system : str
            System prompt. Prepended as the first message.
        max_tokens : int, optional
            Upper bound on generated tokens.

        Returns
        -------
        LLMResponse
            Normalised response with stop reason, text, tool calls, and
            token usage (``prompt_tokens``/``completion_tokens`` mapped to
            ``input_tokens``/``output_tokens``).
        """
        oa_messages: list[dict] = [{"role": "system", "content": system}, *messages]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oa_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        usage = getattr(response, "usage", None)

        return LLMResponse(
            stop_reason=stop_reason,
            text_content=message.content,
            tool_calls=tool_calls,
            raw_content=message,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )

    def build_tool_result_messages(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Emit one ``role="tool"`` message per tool result.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            ``(tool_call_id, output)`` pairs.

        Returns
        -------
        list[dict]
            One ``role="tool"`` message per pair, in order.
        """
        return [{"role": "tool", "tool_call_id": tid, "content": out} for tid, out in pairs]

    def format_assistant_message(self, raw_content: Any) -> dict:
        """Rebuild the ``assistant`` message from a prior response.

        Drops empty ``content`` (OpenAI accepts messages with only
        ``tool_calls``) and re-encodes tool calls in the wire format the
        Chat Completions API expects.

        Parameters
        ----------
        raw_content : Any
            ``LLMResponse.raw_content`` from a prior :meth:`create_message`.

        Returns
        -------
        dict
            A single assistant message ready to append to ``messages``.
        """
        msg: dict[str, Any] = {"role": "assistant"}
        if getattr(raw_content, "content", None):
            msg["content"] = raw_content.content
        if getattr(raw_content, "tool_calls", None):
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_content.tool_calls
            ]
        return msg
