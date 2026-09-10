"""Typed tool registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """Declarative tool with JSON-schema parameters and a Python handler."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    handler: Callable[..., Any]
    side_effect: bool = False
    requires_approval: bool = False
    cost: float = 0.0
    is_approval: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "side_effect": self.side_effect,
            "requires_approval": self.requires_approval,
            "is_approval": self.is_approval,
            "cost": self.cost,
        }


class ToolRegistry:
    """Name → ToolSpec map with schema export and invocation."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.get(name)
        return tool.handler(**arguments)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools
