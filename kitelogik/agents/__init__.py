# SPDX-License-Identifier: Apache-2.0
"""
Public API for :mod:`kitelogik.agents`.

Core:

- :class:`AgentSession` — the orchestrator.
- :class:`SessionResult` — the return type.
- :data:`DEFAULT_SYSTEM_PROMPT` — the governance-aware default prompt.
- :func:`default_memory_write_policy` — default trust-tier classifier.

LLM provider clients — all implement :class:`~kitelogik.agents.llm.LLMClient`:

- :class:`~kitelogik.agents.llm.AnthropicLLMClient` — default, no extras.
- :class:`~kitelogik.agents.openai_client.OpenAIClient` — ``pip install 'kitelogik[openai]'``.
- :class:`~kitelogik.agents.google_client.GoogleClient` — ``pip install 'kitelogik[google]'``.

Retry policy:

- :class:`~kitelogik.agents.llm.RetryConfig` — exponential backoff + jitter.
- :func:`~kitelogik.agents.llm.is_retryable_error` — status-code classifier.

Errors: :class:`AgentSessionError` (base), :class:`LLMProviderError`,
:class:`ToolHandlerError`, :class:`SessionAlreadyRanError`.
:class:`~kitelogik.governed.GovernanceError` is re-exported from
:mod:`kitelogik.agents.errors` as a sibling.
"""

from .errors import (
    AgentSessionError,
    LLMProviderError,
    SessionAlreadyRanError,
    ToolHandlerError,
)
from .llm import AnthropicLLMClient, RetryConfig, is_retryable_error
from .session import (
    DEFAULT_SYSTEM_PROMPT,
    AgentSession,
    SessionResult,
    default_memory_write_policy,
)

__all__ = [
    "AgentSession",
    "AgentSessionError",
    "AnthropicLLMClient",
    "DEFAULT_SYSTEM_PROMPT",
    "LLMProviderError",
    "RetryConfig",
    "SessionAlreadyRanError",
    "SessionResult",
    "ToolHandlerError",
    "default_memory_write_policy",
    "is_retryable_error",
]
