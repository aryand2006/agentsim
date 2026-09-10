"""agentsim — deterministic seeded fault simulation for tool-using AI agents."""

from __future__ import annotations

from agentsim.contracts.base import (
    AllowedTools,
    ApprovalBeforeSideEffect,
    Contract,
    ForbiddenTools,
    MaxCost,
    MaxSteps,
    RequiredToolSequence,
)
from agentsim.faults.injector import FaultCampaign, FaultInjector, FaultKind
from agentsim.models import (
    AgentAction,
    Observation,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryStep,
)
from agentsim.registry import ToolRegistry, ToolSpec
from agentsim.result import SimResult, Verdict, Violation
from agentsim.runner import SimConfig, Simulator
from agentsim.shrink import shrink_trajectory

__all__ = [
    "AllowedTools",
    "ApprovalBeforeSideEffect",
    "AgentAction",
    "Contract",
    "FaultCampaign",
    "FaultInjector",
    "FaultKind",
    "ForbiddenTools",
    "MaxCost",
    "MaxSteps",
    "Observation",
    "RequiredToolSequence",
    "SimConfig",
    "SimResult",
    "Simulator",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "Trajectory",
    "TrajectoryStep",
    "Verdict",
    "Violation",
    "shrink_trajectory",
    "__version__",
]

__version__ = "0.1.0"
