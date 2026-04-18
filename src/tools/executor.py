from __future__ import annotations

import json
from typing import Any

from src.skills.registry import SkillRegistry, get_registry
from src.config import TOOL_CALL_MAX_ITERATIONS


class ToolExecutor:
    def __init__(self, registry: SkillRegistry | None = None):
        self.registry = registry or get_registry()

    def execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        func = tool_call.get("function", {})
        name = func.get("name", "")
        arguments_str = func.get("arguments", "{}")
        tool_call_id = tool_call.get("id", "")

        try:
            if isinstance(arguments_str, str):
                arguments = json.loads(arguments_str)
            else:
                arguments = arguments_str
        except json.JSONDecodeError:
            arguments = {}

        try:
            result = self.registry.execute(name, **arguments)
        except ValueError as exc:
            result = f"Error: {exc}"
        except Exception as exc:
            result = f"Error executing tool '{name}': {exc}"

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }

    def process_tool_calls(
        self,
        message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        tool_calls = message.get("tool_calls", [])
        results = []
        for tc in tool_calls:
            result_msg = self.execute_tool_call(tc)
            results.append(result_msg)
        return results

    def get_tools_schema(self) -> list[dict[str, Any]]:
        return self.registry.get_openai_tools()
