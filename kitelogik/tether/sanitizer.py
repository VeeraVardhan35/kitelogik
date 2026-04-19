# SPDX-License-Identifier: Apache-2.0
import re
import unicodedata
from typing import Any

from .models import SanitizedResponse

# Unicode characters that adversaries may use to bypass \s+ matching.
# Includes zero-width spaces, non-breaking spaces, and other invisible separators.
_UNICODE_WHITESPACE = re.compile(
    "[\u00a0\u200b\u200c\u200d\u2028\u2029\ufeff\u2060\u180e\u202f\u205f\u3000]"
)

# Unicode tag characters (U+E0000–U+E007F) are invisible and can smuggle
# instructions past both humans and regex scanners. The ASCII-subset range
# U+E0020–U+E007E mirrors printable ASCII (space through tilde) as invisible
# glyphs — a known exfiltration / prompt-injection vector. We demirror these
# back to their ASCII equivalents so the injection scanner sees the real text.
_TAG_ASCII_MIRROR_START = 0xE0020
_TAG_ASCII_MIRROR_END = 0xE007E
# Remaining tag codepoints (language tag, cancel tag, other non-mirrored) carry
# no legitimate semantic content in tool output and are stripped.
_UNICODE_TAG_NONASCII = re.compile(
    "[\U000e0000-\U000e001f\U000e007f]",
)


def _demirror_unicode_tags(text: str) -> str:
    """Map U+E0020..U+E007E tag chars to their ASCII equivalents in-place."""
    if not any(_TAG_ASCII_MIRROR_START <= ord(ch) <= _TAG_ASCII_MIRROR_END for ch in text):
        return text
    return "".join(
        chr(ord(ch) - 0xE0000)
        if _TAG_ASCII_MIRROR_START <= ord(ch) <= _TAG_ASCII_MIRROR_END
        else ch
        for ch in text
    )


def _normalize_for_scan(text: str) -> str:
    """
    Normalise text before injection scanning.

    Converts confusable unicode to ASCII equivalents (NFKC), demirrors
    invisible unicode tag characters back to their ASCII counterparts,
    strips the remaining non-mirrored tag codepoints, and replaces
    invisible whitespace/joiners with regular spaces so that patterns
    like ``ignore\\s+previous\\s+instructions`` cannot be bypassed with
    zero-width or tag characters.
    """
    # NFKC normalisation maps full-width and other confusable characters
    # to their standard equivalents (e.g. fullwidth 'A' → 'A').
    text = unicodedata.normalize("NFKC", text)
    # Demirror tag chars that mirror printable ASCII (U+E0020..U+E007E).
    text = _demirror_unicode_tags(text)
    # Strip remaining (non-mirrored) tag codepoints.
    text = _UNICODE_TAG_NONASCII.sub("", text)
    # Replace invisible whitespace variants with regular spaces.
    text = _UNICODE_WHITESPACE.sub(" ", text)
    return text


# Ordered list of (regex_pattern, label) pairs.
# Patterns are case-insensitive and anchored to detect injection attempts
# embedded anywhere in tool output content.
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?previous\s+instructions?", "ignore_previous_instructions"),
    (r"disregard\s+(your\s+)?(previous\s+)?instructions?", "disregard_instructions"),
    # "you are now" followed by something that strips restrictions
    (
        r"you\s+are\s+now\s+(an?\s+)?(?:unrestricted|unfiltered|jailbroken)",
        "you_are_now_unrestricted",
    ),
    (r"new\s+instructions?\s*:", "new_instructions"),
    (r"\[SYSTEM\]", "system_marker"),
    (r"<\s*/?instructions?\s*>", "instructions_tag"),
    (
        r"override\s+(all\s+)?(?:safety|security|policy)\s+(?:rules?|constraints?|measures?)",
        "policy_override",
    ),
    (r"act\s+as\s+if\s+you\s+(?:have\s+no|are\s+without)\s+restrictions?", "act_as_unrestricted"),
    (
        r"(?:print|show|output|reveal|display)\s+(your\s+)?(system\s+)?(?:prompt|instructions?)",
        "prompt_extraction",
    ),
    (
        r"forget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?(?:instructions?|training|guidelines?)",
        "forget_instructions",
    ),
    # Role-confusion attempts — instruct the model to assume a persona that
    # would strip its guardrails. Distinct from "you are now unrestricted"
    # in that these push a specific impersonation target.
    (
        r"(?:assume|pretend)\s+(?:that\s+)?you\s+are\s+(?:an?\s+)?(?:system\s+administrator|root|developer|human|admin)",
        "role_confusion_assume",
    ),
    (
        r"(?:act|behave)\s+as\s+(?:an?\s+)?(?:system\s+administrator|root|developer|human|admin)",
        "role_confusion_act_as",
    ),
    (
        r"in\s+the\s+role\s+of\s+(?:an?\s+)?(?:system\s+administrator|root|developer|human|admin)",
        "role_confusion_in_the_role_of",
    ),
    (
        r"if\s+you\s+(?:were|are)\s+(?:an?\s+)?(?:system\s+administrator|root|developer|human|admin)",
        "role_confusion_if_you_were",
    ),
]

_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), label) for pattern, label in _INJECTION_PATTERNS
]


def sanitize_tool_output(content: str) -> SanitizedResponse:
    """Scan tool output for embedded prompt injection payloads and redact them.

    Primary defence against indirect prompt injection — malicious
    instructions embedded in data the agent reads (web pages, documents,
    database records, MCP server responses).

    Parameters
    ----------
    content : str
            Raw tool output to scan.

    Returns
    -------
    SanitizedResponse
            Sanitized content with modification flag and matched patterns.
    """
    if not content:
        return SanitizedResponse(content=content, was_modified=False)

    # Normalise to defeat unicode whitespace / confusable-character bypasses.
    modified = _normalize_for_scan(content)
    patterns_found: list[str] = []

    for compiled_pattern, label in _COMPILED:
        if compiled_pattern.search(modified):
            patterns_found.append(label)
            modified = compiled_pattern.sub("[REDACTED]", modified)

    return SanitizedResponse(
        content=modified,
        was_modified=bool(patterns_found),
        injection_patterns_found=patterns_found,
    )


def sanitize_tool_schema(schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Scan an MCP tool schema for injection payloads embedded in metadata.

    MCP servers return tool definitions with ``name`` and ``description``
    fields that flow directly into the agent's system prompt when the
    schema is handed to the LLM. A compromised or malicious MCP server
    can embed injection payloads in either field — the agent never sees
    them as "tool output" (so ``sanitize_tool_output`` never fires), but
    the LLM treats them as trusted framing.

    This helper returns a shallow-copied schema with ``name`` and
    ``description`` run through the same scanner, plus the list of
    injection labels found (empty if clean). Call this before passing
    any externally-sourced tool schema into ``AgentSession(tools=...)``
    or any framework adapter.

    Parameters
    ----------
    schema : dict
            An MCP tool schema (``{"name": str, "description": str, ...}``).

    Returns
    -------
    tuple[dict, list[str]]
            Sanitized schema and the accumulated injection labels.
    """
    sanitized = dict(schema)
    patterns_found: list[str] = []

    for field in ("name", "description"):
        value = sanitized.get(field)
        if isinstance(value, str) and value:
            result = sanitize_tool_output(value)
            sanitized[field] = result.content
            patterns_found.extend(result.injection_patterns_found)

    return sanitized, patterns_found
