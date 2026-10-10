"""Name -> Tool lookup for the agent loop and the control service."""
from collections.abc import Iterable
from typing import Any

from marketplace_agent.tools.base import Tool
from marketplace_agent.tools.impl import TOOL_CLASSES


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]


def default_registry() -> ToolRegistry:
    return ToolRegistry(cls() for cls in TOOL_CLASSES)
