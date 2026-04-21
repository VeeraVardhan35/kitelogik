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

# Bidi / RTL-override controls (U+202A..U+202E and U+2066..U+2069). These
# reorder text visually without changing codepoint order, letting an attacker
# display "safe" text while smuggling injection payloads in the logical
# stream. They have no legitimate use in structured tool output; strip them.
_BIDI_CONTROLS = re.compile("[\u202a-\u202e\u2066-\u2069]")

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

# Hand-curated homoglyph fold — Cyrillic / Greek / Latin-extended lookalikes
# that NFKC does not normalise because they belong to distinct scripts. Keeps
# the table small (common attack chars only); covers "ignоre previous
# instructions" style bypasses where a single Latin letter is substituted
# with a visually identical Cyrillic codepoint. Values are ASCII.
_CONFUSABLES: dict[str, str] = {
    # Cyrillic lowercase → Latin lowercase
    "а": "a",
    "в": "B",
    "с": "c",
    "е": "e",
    "һ": "h",
    "і": "i",
    "ј": "j",
    "ӏ": "l",
    "м": "M",
    "о": "o",
    "р": "p",
    "ԛ": "q",
    "ѕ": "s",
    "т": "T",
    "у": "y",
    "х": "x",
    # Cyrillic uppercase → Latin
    "А": "A",
    "В": "B",
    "С": "C",
    "Е": "E",
    "Н": "H",
    "І": "I",
    "Ј": "J",
    "К": "K",
    "М": "M",
    "О": "O",
    "Р": "P",
    "Ѕ": "S",
    "Т": "T",
    "У": "Y",
    "Х": "X",
    # Greek → Latin (visually identical in most fonts)
    "α": "a",
    "ο": "o",
    "ρ": "p",
    "ν": "v",
    "τ": "T",
    "ι": "i",
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Χ": "X",
    "Υ": "Y",
    "Ζ": "Z",
}
# Compiled character class for fast dispatch; falls back to table lookup
# only when a confusable is actually present in the input.
_CONFUSABLE_CHARS = "".join(_CONFUSABLES.keys())


def _fold_confusables(text: str) -> str:
    """Replace Cyrillic/Greek lookalikes with their Latin ASCII equivalents."""
    if not any(ch in _CONFUSABLES for ch in text):
        return text
    return "".join(_CONFUSABLES.get(ch, ch) for ch in text)


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
    # to their standard equivalents (e.g. fullwidth 'A' → 'A'). Math
    # alphanumeric, script, and double-struck ranges are all NFKC-folded.
    text = unicodedata.normalize("NFKC", text)
    # Strip bidi / RTL override controls (no legitimate use in tool output).
    text = _BIDI_CONTROLS.sub("", text)
    # Fold Cyrillic / Greek lookalikes to their Latin ASCII twins so payloads
    # like "ignоre previous instructions" (Cyrillic 'о') match the regex.
    text = _fold_confusables(text)
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
#
# The "ignore X previous Y" family (ignore / disregard / forget / skip /
# bypass / cancel / override) is consolidated into shared sub-patterns
# below. Each verb family follows the shape:
#
#   <verb> + <qualifier>{0,2} + <temporal> + <instruction-noun>      (soft)
#   <verb> + <qualifier>{0,2} + <temporal>? + <instruction-noun>     (strong)
#
# Temporal is REQUIRED for the soft verbs ("ignore", "skip", "bypass",
# "cancel", "override") — without it, "ignore the spam" would match. It
# is OPTIONAL for the strong verbs ("disregard", "forget") whose verb
# alone already implies an instruction context. Both the temporal
# alternation and the noun alternation are flat (no nested quantifiers),
# so the regex engine runs in linear time on any input.
# Zero, one, or two stacked qualifiers (e.g. "all your", "the previous")
# — bounded to prevent catastrophic backtracking.
_QUALIFIER = r"(?:(?:all|the|your|any)\s+){0,2}"
_TEMPORAL = r"(?:previous|prior|earlier|above|preceding|foregoing)"
# Target nouns kept deliberately narrow — only words that describe
# meta-instruction in an agent/LLM context. Generic nouns like "data",
# "steps", "emails", "orders" (e-commerce), "directions" (navigation) are
# excluded because they collide with legitimate business data. Attackers
# phrasing an attack in those forms can trivially substitute to
# "instructions" / "rules" instead.
_INSTRUCTION_NOUN = (
    r"(?:instructions?|rules?|guidance|guidelines?|directives?|prompts?)"
)

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # Temporal qualifier REQUIRED for soft verbs — guards against false
    # positives on phrases like "ignore the spam folder".
    (
        rf"ignore\s+{_QUALIFIER}{_TEMPORAL}\s+{_INSTRUCTION_NOUN}",
        "ignore_previous_instructions",
    ),
    # Temporal qualifier OPTIONAL for strong verbs — "disregard instructions"
    # on its own is already suspicious enough to redact.
    (
        rf"disregard\s+{_QUALIFIER}(?:{_TEMPORAL}\s+)?{_INSTRUCTION_NOUN}",
        "disregard_instructions",
    ),
    (
        rf"forget\s+{_QUALIFIER}(?:{_TEMPORAL}\s+)?(?:{_INSTRUCTION_NOUN}|training)",
        "forget_instructions",
    ),
    # skip / bypass / cancel / override + temporal + target.
    (
        rf"(?:skip|bypass|cancel|override)\s+{_QUALIFIER}{_TEMPORAL}\s+{_INSTRUCTION_NOUN}",
        "override_previous_instructions",
    ),
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
    # Markdown / HTML URI injection — embeds executable schemes in link or
    # image syntax that an agent rendering the response might follow, or
    # that a downstream UI might render as a live link.
    (r"\]\(\s*javascript\s*:", "markdown_javascript_uri"),
    (r"\]\(\s*data\s*:\s*text/html", "markdown_data_html_uri"),
    (r"<\s*/?\s*(?:system|sys|assistant|user|developer)\s*>", "fake_role_tag"),
    # Multilingual variants of "ignore previous instructions". Narrow
    # wording on purpose — kept to single-phrase forms of the top attack,
    # so false positives stay low in multilingual business data.
    (r"ignora\s+(?:las\s+)?instrucciones\s+(?:anteriores|previas)", "ignore_previous_es"),
    (
        r"ignorez\s+(?:les\s+)?instructions\s+(?:pr[eé]c[eé]dentes|ant[eé]rieures)",
        "ignore_previous_fr",
    ),
    (r"ignorieren?\s+sie\s+(?:die\s+)?(?:vorherigen|vorigen)\s+anweisungen", "ignore_previous_de"),
    (r"忽略(?:之前|以前|先前)(?:的)?指[令示]", "ignore_previous_zh"),
    (r"تجاهل\s+(?:جميع\s+)?التعليمات\s+السابقة", "ignore_previous_ar"),
    (r"पिछले\s+निर्देशों\s+को\s+अनदेखा", "ignore_previous_hi"),
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
