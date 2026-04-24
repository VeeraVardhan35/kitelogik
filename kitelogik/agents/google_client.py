# SPDX-License-Identifier: Apache-2.0
"""
Google Gemini implementation of the LLMClient protocol.

Install the optional extra::

    pip install 'kitelogik[google]'

Then pass a ``GoogleClient`` to ``AgentSession``::

    from kitelogik.agents.google_client import GoogleClient

    session = AgentSession(gate=gate, context=ctx, llm_client=GoogleClient())

Tool schemas passed via ``tools=`` must follow Gemini's function-declaration
format (a list of ``{"name": ..., "description": ..., "parameters": ...}``
dicts or equivalent ``Tool`` objects from ``google.genai.types``).

Status: experimental. Covers the common tool-use loop; edge cases (grounding,
multi-turn safety settings) are not yet exposed through the protocol.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from kitelogik.agents.llm import DEFAULT_MAX_TOKENS, LLMResponse, ToolCall

logger = logging.getLogger(__name__)


class GoogleClient:
    """:class:`LLMClient` implementation for Google Gemini (``google-genai`` SDK).

    Parameters
    ----------
    api_key : str or None, optional
        Gemini API key. Falls back to ``GEMINI_API_KEY`` or
        ``GOOGLE_API_KEY`` if not provided.
    default_model : str or None, optional
        Model used when :class:`AgentSession` is constructed without an
        explicit ``model=``. Defaults to ``"gemini-2.0-flash"``.

    Raises
    ------
    RuntimeError
        If ``google-genai`` is not installed, or if no API key is provided
        and neither ``GEMINI_API_KEY`` nor ``GOOGLE_API_KEY`` is set.

    Notes
    -----
    Status: experimental. The common tool-use loop is covered; edge cases
    (grounding, multi-turn safety settings, cached contexts) are not yet
    exposed through the protocol.

    Examples
    --------
    >>> from kitelogik.agents.google_client import GoogleClient
    >>> client = GoogleClient()
    >>> session = AgentSession(gate=gate, context=ctx, llm_client=client)
    """

    default_model: str = "gemini-2.0-flash"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
    ) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "google-genai is not installed. Install the optional extra: "
                "`pip install 'kitelogik[google]'` or `pip install 'google-genai>=0.3'`."
            ) from e

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Pass api_key=... or set "
                "GEMINI_API_KEY / GOOGLE_API_KEY in the environment."
            )
        self._client = genai.Client(api_key=key)
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
        """Send a ``generate_content`` request and return a normalised response.

        Parameters
        ----------
        model : str
            Gemini model identifier.
        messages : list[dict]
            Conversation history. Entries already in Gemini shape (with a
            ``parts`` key) pass through; plain ``content`` strings are
            wrapped via :meth:`_messages_to_contents`.
        tools : list[dict]
            Function declarations. Dicts are converted to
            ``types.FunctionDeclaration`` at call time.
        system : str
            System instruction.
        max_tokens : int, optional
            Upper bound on generated tokens (``max_output_tokens``).

        Returns
        -------
        LLMResponse
            Normalised response with stop reason, text, tool calls, and
            token usage from ``usage_metadata``.
        """
        from google.genai import types

        contents = self._messages_to_contents(messages)
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        if tools:
            declarations = [
                t if isinstance(t, types.FunctionDeclaration) else types.FunctionDeclaration(**t)
                for t in tools
            ]
            config_kwargs["tools"] = [types.Tool(function_declarations=declarations)]

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for candidate in response.candidates or []:
            for part in getattr(candidate.content, "parts", []) or []:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc:
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        ToolCall(id=f"gen_{uuid.uuid4().hex[:12]}", name=fc.name, input=args)
                    )

        stop_reason = "tool_use" if tool_calls else "end_turn"
        usage = getattr(response, "usage_metadata", None)

        return LLMResponse(
            stop_reason=stop_reason,
            text_content="".join(text_parts) or None,
            tool_calls=tool_calls,
            raw_content=response,
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
        )

    def build_tool_result_messages(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Emit one ``role="function"`` message per tool result.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            ``(tool_call_id, output)`` pairs. Note Gemini uses the tool *name*
            as identifier, so ``tool_call_id`` here is the tool name carried
            through from the prior :class:`ToolCall`.

        Returns
        -------
        list[dict]
            One ``role="function"`` content block per pair, each holding a
            ``function_response`` part.
        """
        return [
            {
                "role": "function",
                "parts": [
                    {
                        "function_response": {
                            "name": tid,
                            "response": self._parse_maybe_json(out),
                        }
                    }
                ],
            }
            for tid, out in pairs
        ]

    def format_assistant_message(self, raw_content: Any) -> dict:
        """Rebuild the ``model`` turn from a prior Gemini response.

        Parameters
        ----------
        raw_content : Any
            ``LLMResponse.raw_content`` from a prior :meth:`create_message`.

        Returns
        -------
        dict
            A ``role="model"`` content block with text and function-call
            parts, ready to append to ``contents``.
        """
        parts: list[dict] = []
        for candidate in getattr(raw_content, "candidates", []) or []:
            for part in getattr(candidate.content, "parts", []) or []:
                if getattr(part, "text", None):
                    parts.append({"text": part.text})
                fc = getattr(part, "function_call", None)
                if fc:
                    parts.append({"function_call": {"name": fc.name, "args": dict(fc.args or {})}})
        return {"role": "model", "parts": parts}

    @staticmethod
    def _messages_to_contents(messages: list[dict]) -> list[dict]:
        """Translate canonical messages to Gemini ``contents``.

        Messages already in Gemini shape (containing a ``parts`` key) pass
        through untouched. Plain ``content`` text is wrapped in a
        ``parts=[{text: ...}]`` block with role mapping (``assistant`` →
        ``model``).

        Parameters
        ----------
        messages : list[dict]
            Canonical message history.

        Returns
        -------
        list[dict]
            Gemini-shaped content blocks suitable for ``generate_content``.
        """
        contents: list[dict] = []
        for m in messages:
            role = m.get("role", "user")
            if "parts" in m:
                contents.append(m)
                continue
            text = m.get("content")
            if isinstance(text, str):
                contents.append(
                    {
                        "role": "model" if role == "assistant" else role,
                        "parts": [{"text": text}],
                    }
                )
            elif isinstance(text, list):
                contents.append({"role": role, "parts": text})
        return contents

    @staticmethod
    def _parse_maybe_json(s: str) -> dict:
        """Parse ``s`` as JSON if possible, else wrap as ``{"result": s}``.

        Parameters
        ----------
        s : str
            Tool result content.

        Returns
        -------
        dict
            Parsed object if ``s`` is a JSON object; otherwise a single-key
            ``{"result": ...}`` wrapper (required by Gemini's
            ``function_response.response`` schema).
        """
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return {"result": s}
        return parsed if isinstance(parsed, dict) else {"result": parsed}
