# SPDX-License-Identifier: Apache-2.0
"""
Error taxonomy for :class:`~kitelogik.agents.session.AgentSession`.

All exceptions intentionally raised from ``AgentSession.run_async`` /
``run_sync`` inherit from :class:`AgentSessionError`, so callers can catch the
whole family with one handler or pick a specific subclass for
provider-vs-handler-vs-policy failures.

:class:`~kitelogik.governed.GovernanceError` (raised on policy denials) lives
in :mod:`kitelogik.governed` for historical reasons; it is re-exported here
as a sibling so that catching ``(AgentSessionError, GovernanceError)`` covers
every expected failure mode.
"""

from __future__ import annotations

from kitelogik.governed import GovernanceError


class AgentSessionError(Exception):
    """Base class for infrastructure failures raised from ``AgentSession``.

    Does **not** subsume :class:`~kitelogik.governed.GovernanceError` —
    policy denials are domain errors and remain a separate hierarchy.

    Examples
    --------
    Catch any infrastructure failure from a session run::

        try:
            await session.run_async("do the thing")
        except AgentSessionError as e:
            logger.error("session failed: %s", e)
    """


class LLMProviderError(AgentSessionError):
    """The LLM provider returned a non-recoverable error.

    Raised after retries and fallback (when configured) have been exhausted.
    The underlying SDK exception is available both via ``__cause__`` and
    the ``original`` attribute for easy inspection.

    Parameters
    ----------
    message : str
        Human-readable description of what failed.
    original : Exception or None, optional
        The underlying SDK exception. Stored on ``self.original`` and
        chained as ``__cause__``.

    Attributes
    ----------
    original : Exception or None
        The underlying SDK exception, if known. Inspect provider-specific
        fields (``status_code``, rate-limit headers, etc.) here.
    """

    def __init__(self, message: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original


class ToolHandlerError(AgentSessionError):
    """The caller-supplied ``tool_handler`` raised an exception.

    A ``tool_handler`` that raises an :class:`AgentSessionError` subclass
    (e.g. a custom ``ToolHandlerError``) propagates unchanged. Other
    exceptions are wrapped in this class so the session's error contract
    stays stable.

    Parameters
    ----------
    tool_name : str
        Name of the tool whose handler raised.
    message : str
        Human-readable description.
    original : Exception or None, optional
        The underlying handler exception.

    Attributes
    ----------
    tool_name : str
        The tool that failed.
    original : Exception or None
        The original exception raised by the handler.
    """

    def __init__(self, tool_name: str, message: str, original: Exception | None = None) -> None:
        super().__init__(f"tool_handler for '{tool_name}' failed: {message}")
        self.tool_name = tool_name
        self.original = original


class SessionAlreadyRanError(AgentSessionError):
    """Raised on a second ``run_async`` / ``run_sync`` call on the same session.

    :class:`~kitelogik.agents.session.AgentSession` is single-use by design:
    each run mutates session-scoped state (credentials, context counters).
    Construct a new session for each run.

    Examples
    --------
    >>> session = AgentSession(gate=gate, context=ctx, llm_client=llm)
    >>> await session.run_async("first question")
    >>> await session.run_async("second question")  # raises
    Traceback (most recent call last):
        ...
    SessionAlreadyRanError: AgentSession instances are single-use. ...
    """


__all__ = [
    "AgentSessionError",
    "GovernanceError",
    "LLMProviderError",
    "ToolHandlerError",
    "SessionAlreadyRanError",
]
