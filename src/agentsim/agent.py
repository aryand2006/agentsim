"""Agent protocol — any callable / object that maps Observation → AgentAction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentsim.models import AgentAction, Observation


@runtime_checkable
class Agent(Protocol):
    """Tool-using agent interface used by the simulator."""

    def step(self, observation: Observation) -> AgentAction:
        """Consume an observation and emit tool calls or a final answer."""
        ...
