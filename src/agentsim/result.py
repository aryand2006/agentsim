"""Simulation verdict, violations, and one-command repro strings."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class ViolationCode(StrEnum):
    REQUIRED_SEQUENCE = "REQUIRED_SEQUENCE"
    ALLOWED_TOOLS = "ALLOWED_TOOLS"
    FORBIDDEN_TOOL = "FORBIDDEN_TOOL"
    MAX_COST = "MAX_COST"
    MAX_STEPS = "MAX_STEPS"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    TOOL_ERROR = "TOOL_ERROR"
    BUDGET_BLOWUP = "BUDGET_BLOWUP"
    AGENT_EXCEPTION = "AGENT_EXCEPTION"


class Violation(BaseModel):
    code: ViolationCode | str
    message: str
    step_index: int
    witness: dict[str, Any] = Field(default_factory=dict)


class SimResult(BaseModel):
    """PASS/FAIL report with seed and one-command repro."""

    verdict: Verdict
    seed: int
    violations: list[Violation] = Field(default_factory=list)
    steps: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    trajectory_path: str | None = None
    faults_applied: list[str] = Field(default_factory=list)
    final_answer: str | None = None
    repro: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == Verdict.PASS

    def summarize(self) -> str:
        lines = [
            f"[{self.verdict.value}] seed={self.seed} steps={self.steps} "
            f"cost={self.total_cost:.4f} tokens={self.total_tokens}"
        ]
        if self.faults_applied:
            lines.append(f"  faults: {', '.join(self.faults_applied)}")
        for v in self.violations:
            lines.append(
                f"  FAIL@{v.step_index} {v.code}: {v.message}"
            )
        if self.repro:
            lines.append(f"  repro: {self.repro}")
        return "\n".join(lines)

    @classmethod
    def make_repro(
        cls,
        *,
        seed: int,
        campaign: str | None = None,
        trajectory: str | None = None,
        contracts: str | None = None,
    ) -> str:
        parts = ["agentsim run", f"--seed {seed}"]
        if campaign:
            parts.append(f"--campaign {campaign}")
        if trajectory:
            parts.append(f"--replay {trajectory}")
        if contracts:
            parts.append(f"--contracts {contracts}")
        return " ".join(parts)
