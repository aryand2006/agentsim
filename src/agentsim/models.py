"""Core domain models for agent simulation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionKind(StrEnum):
    TOOL_CALLS = "tool_calls"
    FINAL = "final"


class ToolCall(BaseModel):
    """A single tool invocation requested by the agent."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""


class ToolResult(BaseModel):
    """Result returned for a tool call (possibly after fault injection)."""

    call_id: str
    name: str
    ok: bool = True
    output: Any = None
    error: str | None = None
    dropped: bool = False
    timed_out: bool = False
    latency_ms: float = 0.0
    cost: float = 0.0
    fault: str | None = None


class AgentAction(BaseModel):
    """What the agent emits after an observation."""

    kind: Literal["tool_calls", "final"]
    tool_calls: list[ToolCall] = Field(default_factory=list)
    answer: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0


class Observation(BaseModel):
    """Input presented to the agent at a given step."""

    step: int
    prompt: str = ""
    tool_results: list[ToolResult] = Field(default_factory=list)
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    """One recorded interaction turn."""

    index: int
    observation: Observation
    action: AgentAction
    tool_results: list[ToolResult] = Field(default_factory=list)
    faults_applied: list[str] = Field(default_factory=list)
    cost: float = 0.0
    cumulative_cost: float = 0.0


class Trajectory(BaseModel):
    """Full recorded agent run, serializable to JSON."""

    seed: int
    prompt: str = ""
    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_answer: str | None = None
    total_cost: float = 0.0
    total_tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, *, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str | bytes) -> Trajectory:
        return cls.model_validate_json(data)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> Trajectory:
        with open(path, encoding="utf-8") as fh:
            return cls.from_json(fh.read())
