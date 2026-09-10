"""Delta-debug style shrinking of failing trajectories (stretch: best-effort)."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from agentsim.models import Trajectory, TrajectoryStep
from agentsim.registry import ToolRegistry
from agentsim.result import SimResult, Verdict
from agentsim.runner import SimConfig, Simulator


def shrink_trajectory(
    trajectory: Trajectory,
    registry: ToolRegistry,
    config: SimConfig,
    *,
    still_fails: Callable[[SimResult], bool] | None = None,
) -> tuple[Trajectory, SimResult]:
    """Minimize a failing trace while preserving the failure.

    Strategy (ddmin-inspired):
    1. Drop trailing steps after the first violation when possible.
    2. Try removing individual intermediate tool-call steps.
    3. Try removing individual tool calls within a multi-call step.

    This is a best-effort shrink — not guaranteed minimal for all campaigns
    because faults are seeded against the *run*, not step-local RNGs.
    """

    def fails(result: SimResult) -> bool:
        if still_fails is not None:
            return still_fails(result)
        return result.verdict == Verdict.FAIL

    sim = Simulator(registry, config)
    baseline = sim.replay(trajectory)
    if not fails(baseline):
        return trajectory, baseline

    best = deepcopy(trajectory)
    best_result = baseline

    # 1) Truncate after first violation step when present.
    if baseline.violations:
        cut = baseline.violations[0].step_index + 1
        if 0 < cut < len(best.steps):
            candidate = deepcopy(best)
            candidate.steps = candidate.steps[:cut]
            # Ensure last action can terminate.
            if candidate.steps and candidate.steps[-1].action.kind != "final":
                # Keep as-is; replay will stop on violation or max steps.
                pass
            result = sim.replay(candidate)
            if fails(result):
                best, best_result = candidate, result

    # 2) Remove one step at a time (skip first observation-bearing steps carefully).
    changed = True
    while changed:
        changed = False
        for i in range(len(best.steps)):
            # Never remove the sole step.
            if len(best.steps) <= 1:
                break
            candidate = deepcopy(best)
            del candidate.steps[i]
            _reindex(candidate)
            result = sim.replay(candidate)
            if fails(result) and len(candidate.steps) < len(best.steps):
                best, best_result = candidate, result
                changed = True
                break

    # 3) Thin multi-tool batches.
    for si, step in enumerate(best.steps):
        calls = step.action.tool_calls
        if len(calls) <= 1:
            continue
        for ci in range(len(calls)):
            candidate = deepcopy(best)
            new_calls = [
                c
                for j, c in enumerate(candidate.steps[si].action.tool_calls)
                if j != ci
            ]
            if not new_calls:
                continue
            candidate.steps[si].action.tool_calls = new_calls
            result = sim.replay(candidate)
            if fails(result):
                best, best_result = candidate, result
                break

    return best, best_result


def _reindex(trajectory: Trajectory) -> None:
    new_steps: list[TrajectoryStep] = []
    for i, step in enumerate(trajectory.steps):
        new_steps.append(step.model_copy(update={"index": i}))
    trajectory.steps = new_steps
