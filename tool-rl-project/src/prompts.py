"""Prompt construction helpers for chat + tool usage."""
from __future__ import annotations

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

    raise NotImplementedError("Implement conversation templating")


def render_tool_call(tool_name: str, arguments_json: str) -> str:
    """Return the canonical tool-call block used as the target sequence."""

    raise NotImplementedError("Implement structured block rendering")
