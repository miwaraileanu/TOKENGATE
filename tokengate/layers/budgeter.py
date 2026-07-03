from __future__ import annotations
import re

from tokengate.core.context import LayerContext, LayerDecision

# ── Request type detection ────────────────────────────────────────────────────

_EXTRACTION_RE = re.compile(
    r"\b(json|extract|list only|output only|parse|return only)\b",
    re.IGNORECASE,
)

# Code: fenced block OR imperative verb within 10 words of a code noun
_CODE_FENCE_RE = re.compile(r"```")
_CODE_VERB_RE = re.compile(r"\b(write|implement|fix|refactor|debug)\b", re.IGNORECASE)
_CODE_NOUN_RE = re.compile(
    r"\b(function|class|script|program|method|api|endpoint)\b", re.IGNORECASE
)

_LONG_FORM_RE = re.compile(
    r"\b(essay|blog|article|explain in detail|write a detailed|comprehensive)\b",
    re.IGNORECASE,
)

_CONFLICT_RE = re.compile(r"\b(detailed|thorough|explain fully|verbose)\b", re.IGNORECASE)


def _has_code_verb_near_noun(text: str) -> bool:
    """True if an imperative code verb appears within 10 words of a code noun."""
    words = text.split()
    for i, word in enumerate(words):
        if _CODE_VERB_RE.match(word):
            window = words[max(0, i - 10): i + 11]
            if any(_CODE_NOUN_RE.match(w) for w in window):
                return True
    return False


def _detect_type(user_text: str) -> tuple[str, str]:
    """Return (type, trigger). First match wins: extraction > code > long_form > chat."""
    if _EXTRACTION_RE.search(user_text):
        m = _EXTRACTION_RE.search(user_text)
        return "extraction", m.group(0) if m else "extraction"

    if _CODE_FENCE_RE.search(user_text):
        return "code", "fenced_block"

    if _has_code_verb_near_noun(user_text):
        m = _CODE_VERB_RE.search(user_text)
        return "code", m.group(0) if m else "code_verb"

    if _LONG_FORM_RE.search(user_text):
        m = _LONG_FORM_RE.search(user_text)
        return "long_form", m.group(0) if m else "long_form"

    return "chat", "default"


def _user_text(req) -> str:
    """Concatenate all user-role message content."""
    parts: list[str] = []
    for msg in req.messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
    return " ".join(parts)


# ── Extraction instruction injection ─────────────────────────────────────────

def _inject_extraction_instruction(req, settings) -> tuple[bool, str | None]:
    """
    Inject the extraction instruction into the system prompt.
    Returns (instruction_added, skip_reason).
    """
    if not settings.budget_extraction_instruction_enabled:
        return False, "disabled"

    instruction = settings.budget_extraction_instruction
    existing_system: dict | None = None
    for msg in req.messages:
        if msg.get("role") == "system":
            existing_system = msg
            break

    if existing_system is None:
        req.messages.insert(0, {"role": "system", "content": instruction})
        return True, None

    current_content = existing_system.get("content", "")
    if _CONFLICT_RE.search(current_content):
        return False, "conflict_directive"

    existing_system["content"] = current_content + "\n" + instruction
    return True, None


# ── Main apply ────────────────────────────────────────────────────────────────

async def apply(ctx: LayerContext) -> LayerContext:
    req = ctx.request
    settings = ctx.settings

    # Skip if client already set max_tokens
    if req.max_tokens is not None:
        ctx.decisions.append(LayerDecision("budgeter", "skip", {"reason": "client_set_max_tokens"}))
        return ctx

    user_text = _user_text(req)
    detected_type, trigger = _detect_type(user_text)

    # Low-confidence skip: very short prompt and not extraction
    word_count = len(user_text.split())
    if word_count < 10 and detected_type != "extraction":
        ctx.decisions.append(LayerDecision("budgeter", "skip", {"reason": "type_uncertain"}))
        return ctx

    # Inject max_tokens
    injected = settings.budget_table[detected_type]
    req.max_tokens = injected

    # Extraction instruction (extraction type only)
    instruction_added = False
    instruction_skip_reason: str | None = None
    if detected_type == "extraction":
        instruction_added, instruction_skip_reason = _inject_extraction_instruction(req, settings)

    ctx.decisions.append(LayerDecision("budgeter", "applied", {
        "type": detected_type,
        "trigger": trigger,
        "max_tokens": injected,
        "instruction_added": instruction_added,
        "instruction_skip_reason": instruction_skip_reason,
    }))
    return ctx
