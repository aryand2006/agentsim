"""Contract assertions on tool sequences, costs, and approvals."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentsim.models import AgentAction, Trajectory, TrajectoryStep
from agentsim.registry import ToolRegistry
from agentsim.result import Violation, ViolationCode


@runtime_checkable
class Contract(Protocol):
    """A check evaluated after each step and/or at end of run."""

    name: str

    def check_step(
        self,
        step: TrajectoryStep,
        history: list[TrajectoryStep],
        registry: ToolRegistry,
    ) -> Violation | None: ...

    def check_end(
        self,
        trajectory: Trajectory,
        registry: ToolRegistry,
    ) -> Violation | None: ...


class _ContractBase:
    name: str = "contract"

    def check_step(
        self,
        step: TrajectoryStep,
        history: list[TrajectoryStep],
        registry: ToolRegistry,
    ) -> Violation | None:
        return None

    def check_end(
        self,
        trajectory: Trajectory,
        registry: ToolRegistry,
    ) -> Violation | None:
        return None


class RequiredToolSequence(_ContractBase):
    """Tool names must appear in order (not necessarily contiguous)."""

    name = "required_tool_sequence"

    def __init__(self, sequence: list[str]) -> None:
        self.sequence = list(sequence)

    def check_end(
        self,
        trajectory: Trajectory,
        registry: ToolRegistry,
    ) -> Violation | None:
        seen = [
            tc.name
            for step in trajectory.steps
            for tc in step.action.tool_calls
        ]
        cursor = 0
        for name in seen:
            if cursor < len(self.sequence) and name == self.sequence[cursor]:
                cursor += 1
        if cursor < len(self.sequence):
            missing = self.sequence[cursor:]
            return Violation(
                code=ViolationCode.REQUIRED_SEQUENCE,
                message=f"missing required sequence suffix: {missing}",
                step_index=len(trajectory.steps) - 1 if trajectory.steps else 0,
                witness={"seen": seen, "required": self.sequence, "missing": missing},
            )
        return None


class AllowedTools(_ContractBase):
    """Only these tool names may be called."""

    name = "allowed_tools"

    def __init__(self, tools: set[str] | list[str]) -> None:
        self.tools = set(tools)

    def check_step(
        self,
        step: TrajectoryStep,
        history: list[TrajectoryStep],
        registry: ToolRegistry,
    ) -> Violation | None:
        for tc in step.action.tool_calls:
            if tc.name not in self.tools:
                return Violation(
                    code=ViolationCode.ALLOWED_TOOLS,
                    message=f"tool {tc.name!r} not in allowed set",
                    step_index=step.index,
                    witness={"tool": tc.name, "allowed": sorted(self.tools)},
                )
        return None


class ForbiddenTools(_ContractBase):
    """These tools must never be called."""

    name = "forbidden_tools"

    def __init__(self, tools: set[str] | list[str]) -> None:
        self.tools = set(tools)

    def check_step(
        self,
        step: TrajectoryStep,
        history: list[TrajectoryStep],
        registry: ToolRegistry,
    ) -> Violation | None:
        for tc in step.action.tool_calls:
            if tc.name in self.tools:
                return Violation(
                    code=ViolationCode.FORBIDDEN_TOOL,
                    message=f"forbidden tool called: {tc.name}",
                    step_index=step.index,
                    witness={"tool": tc.name},
                )
        return None


class MaxCost(_ContractBase):
    """Cumulative cost must stay under a budget."""

    name = "max_cost"

    def __init__(self, limit: float) -> None:
        self.limit = limit

    def check_step(
        self,
        step: TrajectoryStep,
        history: list[TrajectoryStep],
        registry: ToolRegistry,
    ) -> Violation | None:
        if step.cumulative_cost > self.limit:
            return Violation(
                code=ViolationCode.MAX_COST,
                message=f"cost {step.cumulative_cost:.4f} exceeded max {self.limit}",
                step_index=step.index,
                witness={"cost": step.cumulative_cost, "limit": self.limit},
            )
        return None


class MaxSteps(_ContractBase):
    """Hard cap on interaction turns."""

    name = "max_steps"

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def check_step(
        self,
        step: TrajectoryStep,
        history: list[TrajectoryStep],
        registry: ToolRegistry,
    ) -> Violation | None:
        # Step indices are 0-based; exceeding limit-1 means too many steps.
        if step.index + 1 > self.limit:
            return Violation(
                code=ViolationCode.MAX_STEPS,
                message=f"step {step.index + 1} exceeded max {self.limit}",
                step_index=step.index,
                witness={"steps": step.index + 1, "limit": self.limit},
            )
        return None


class ApprovalBeforeSideEffect(_ContractBase):
    """Side-effect tools require a prior approval tool call in the same run."""

    name = "approval_before_side_effect"

    def __init__(self, approval_tools: set[str] | list[str] | None = None) -> None:
        self.approval_tools = set(approval_tools) if approval_tools else None

    def check_step(
        self,
        step: TrajectoryStep,
        history: list[TrajectoryStep],
        registry: ToolRegistry,
    ) -> Violation | None:
        approved = False
        prior = history + [step]
        for s in prior:
            for tc in s.action.tool_calls:
                if self._is_approval(tc.name, registry):
                    approved = True
                if registry.has(tc.name):
                    tool = registry.get(tc.name)
                    if tool.side_effect and not approved:
                        # Approval in the *same* parallel batch does not count
                        # unless it appears before this call in the batch.
                        batch_approved = False
                        for earlier in s.action.tool_calls:
                            if earlier is tc:
                                break
                            if self._is_approval(earlier.name, registry):
                                batch_approved = True
                        # Also accept approval from strictly earlier steps.
                        earlier_ok = any(
                            self._is_approval(tc2.name, registry)
                            for prev in history
                            for tc2 in prev.action.tool_calls
                        )
                        if not (batch_approved or earlier_ok):
                            return Violation(
                                code=ViolationCode.APPROVAL_REQUIRED,
                                message=(
                                    f"side-effect tool {tc.name!r} called "
                                    "without prior approval"
                                ),
                                step_index=step.index,
                                witness={"tool": tc.name},
                            )
        return None

    def _is_approval(self, name: str, registry: ToolRegistry) -> bool:
        if self.approval_tools is not None:
            return name in self.approval_tools
        if not registry.has(name):
            return False
        return registry.get(name).is_approval


def evaluate_contracts(
    contracts: list[Contract],
    step: TrajectoryStep,
    history: list[TrajectoryStep],
    registry: ToolRegistry,
) -> list[Violation]:
    violations: list[Violation] = []
    for c in contracts:
        v = c.check_step(step, history, registry)
        if v is not None:
            violations.append(v)
    return violations


def evaluate_end(
    contracts: list[Contract],
    trajectory: Trajectory,
    registry: ToolRegistry,
) -> list[Violation]:
    violations: list[Violation] = []
    for c in contracts:
        v = c.check_end(trajectory, registry)
        if v is not None:
            violations.append(v)
    return violations


def tool_names_from_action(action: AgentAction) -> list[str]:
    return [tc.name for tc in action.tool_calls]
