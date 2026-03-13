"""Prompt construction helpers for chat + tool usage."""
from __future__ import annotations

import json
from typing import Dict, List

CHAT_TEMPLATE = """
You are a helpful assistant with access to tools. Tools: {tool_list}
Use the following conversation and decide whether to call a tool.
""".strip()

TOOL_CALL_TEMPLATE = """
<assistant>
{{
  "tool_name": "{tool_name}",
  "arguments": {arguments_json}
}}
""".strip()


def render_chat(history: List[Dict[str, str]], tool_list: str) -> str:
    """Render history into a plain-text prompt for decoder-only models."""
    header = CHAT_TEMPLATE.format(tool_list=tool_list)
    turns: List[str] = []
    for item in history:
        role = item.get("role", "user").strip().upper()
        content = item.get("content", "").strip()
        turns.append(f"{role}: {content}")

    conversation = "\n".join(turns)
    if conversation:
        return f"{header}\n\n{conversation}\nASSISTANT:"
    return f"{header}\n\nASSISTANT:"


def render_tool_call(tool_name: str, arguments_json: str) -> str:
    """Return the canonical tool-call block used as the target sequence."""
    normalized_args = arguments_json.strip()
    try:
        parsed = json.loads(normalized_args)
        normalized_args = json.dumps(parsed, ensure_ascii=True)
    except json.JSONDecodeError:
        if not normalized_args:
            normalized_args = "{}"

    return TOOL_CALL_TEMPLATE.format(
        tool_name=tool_name,
        arguments_json=normalized_args,
    )
