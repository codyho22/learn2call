"""Prompt construction helpers for chat + tool usage.

Processed records store the prompt in parts (tool header + conversation turns)
so the prompt can be fitted to a model's context window at tokenization time:
the oldest turns are dropped first, and the target is never truncated.
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence

ROLE_LABELS = {
    "user": "USER",
    "assistant": "ASSISTANT",
    "tool": "TOOL_RESULT",
}
ASSISTANT_PREFIX = "ASSISTANT:"


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    end = text.find(". ")
    if end != -1:
        return text[:end]
    return text.rstrip(".")


def render_tool_signature(tool: Dict) -> str:
    """Render one tool as `name(param: type, ...) - summary`."""
    properties = tool.get("parameters", {}).get("properties", {})
    params = ", ".join(f"{name}: {spec.get('type', 'any')}" for name, spec in properties.items())
    line = f"{tool.get('name', '')}({params})"
    summary = _first_sentence(tool.get("description", ""))
    if summary:
        line += f" - {summary}"
    return line


def render_system(tools: Sequence[Dict]) -> str:
    """Render the tool header shown at the start of every prompt."""
    signatures = "\n".join(render_tool_signature(tool) for tool in tools)
    return f"SYSTEM: You have access to these tools:\n{signatures}\n\n"


def render_turn(turn: Dict[str, str]) -> str:
    label = ROLE_LABELS.get(turn.get("role", "user"), "USER")
    return f"{label}: {turn.get('content', '').strip()}\n"


def render_tool_call(name: str, arguments: Dict) -> str:
    """Return the canonical tool-call JSON used for targets and history."""
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=True)


def render_prompt(system: str, turns: Sequence[Dict[str, str]]) -> str:
    """Render the full (unfitted) prompt as text."""
    return system + "".join(render_turn(turn) for turn in turns) + ASSISTANT_PREFIX


def build_prompt_ids(
    system: str,
    turns: Sequence[Dict[str, str]],
    tokenizer,
    max_prompt_tokens: int,
) -> List[int]:
    """Tokenize a prompt, dropping the oldest turns until it fits.

    If even the newest turn alone does not fit, its oldest tokens are cut.
    """
    def encode(text: str) -> List[int]:
        return tokenizer(text, add_special_tokens=False, verbose=False).input_ids

    system_ids = encode(system)
    suffix_ids = encode(ASSISTANT_PREFIX)
    budget = max_prompt_tokens - len(system_ids) - len(suffix_ids)
    if budget < 0:
        raise ValueError(
            f"Tool header ({len(system_ids)} tokens) does not fit in {max_prompt_tokens} tokens"
        )

    kept: List[List[int]] = []
    used = 0
    for turn in reversed(turns):
        turn_ids = encode(render_turn(turn))
        if used + len(turn_ids) > budget:
            if not kept:
                kept.append(turn_ids[len(turn_ids) - budget:])
            break
        kept.append(turn_ids)
        used += len(turn_ids)

    history_ids = [token for turn_ids in reversed(kept) for token in turn_ids]
    return system_ids + history_ids + suffix_ids


def encode_target(target: str, tokenizer) -> List[int]:
    """Tokenize a target tool call (with the space that follows `ASSISTANT:`)."""
    return tokenizer(" " + target, add_special_tokens=False, verbose=False).input_ids
