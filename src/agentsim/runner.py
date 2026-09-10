"""Deterministic simulator: record live agents or replay fixture trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentsim.agent import Agent
from agentsim.contracts.base import Contract, evaluate_contracts, evaluate_end
from agentsim.faults.injector import FaultCampaign, FaultInjector
from agentsim.models import (
    AgentAction,
    Observation,
    ToolCall,
    Trajectory,
    TrajectoryStep,
)
from agentsim.registry import ToolRegistry
from agentsim.result import SimResult, Verdict, Violation, ViolationCode


@dataclass
class SimConfig:
    seed: int = 0
    max_steps: int = 32
    prompt: str = ""
    campaign: FaultCampaign = field(default_factory=FaultCampaign.none)
    contracts: list[Contract] = field(default_factory=list)
    stop_on_violation: bool = True
    record_dropped_results: bool = True


class FixtureAgent:
    """Deterministic agent that replays recorded actions (no live LLM)."""

    def __init__(self, trajectory: Trajectory) -> None:
        self._actions = [s.action.model_copy(deep=True) for s in trajectory.steps]
        self._i = 0

    def step(self, observation: Observation) -> AgentAction:
        if self._i >= len(self._actions):
            return AgentAction(kind="final", answer="(fixture exhausted)")
        action = self._actions[self._i]
        self._i += 1
        return action


class Simulator:
    """Run an agent against tools with seeded faults and contract checks."""

    def __init__(
        self,
        registry: ToolRegistry,
        config: SimConfig | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or SimConfig()

    def run(self, agent: Agent, *, save_path: str | None = None) -> SimResult:
        injector = FaultInjector(self.config.seed, self.config.campaign)
        trajectory = Trajectory(seed=self.config.seed, prompt=self.config.prompt)
        violations: list[Violation] = []
        history: list[TrajectoryStep] = []
        tool_results: list = []
        cumulative_cost = 0.0
        total_tokens = 0
        final_answer: str | None = None

        for step_idx in range(self.config.max_steps):
            schemas = injector.poison_schemas(self.registry)
            obs = Observation(
                step=step_idx,
                prompt=self.config.prompt,
                tool_results=[
                    r
                    for r in tool_results
                    if self.config.record_dropped_results or not r.dropped
                ],
                tool_schemas=schemas,
            )

            try:
                action = agent.step(obs)
            except Exception as exc:  # noqa: BLE001
                violations.append(
                    Violation(
                        code=ViolationCode.AGENT_EXCEPTION,
                        message=str(exc),
                        step_index=step_idx,
                    )
                )
                break

            # Ensure call IDs for determinism within the step.
            for j, tc in enumerate(action.tool_calls):
                if not tc.call_id:
                    tc.call_id = f"s{step_idx}_c{j}"

            step_faults_before = len(injector.applied)
            tokens_in, tokens_out, blowup_cost = injector.maybe_blowup_tokens(
                action.tokens_in, action.tokens_out
            )
            total_tokens += tokens_in + tokens_out
            cumulative_cost += blowup_cost

            if action.kind == "final" or not action.tool_calls:
                final_answer = action.answer
                step = TrajectoryStep(
                    index=step_idx,
                    observation=obs,
                    action=action,
                    tool_results=[],
                    faults_applied=injector.applied[step_faults_before:],
                    cost=blowup_cost,
                    cumulative_cost=cumulative_cost,
                )
                trajectory.steps.append(step)
                history.append(step)
                violations.extend(
                    evaluate_contracts(
                        self.config.contracts, step, history[:-1], self.registry
                    )
                )
                break

            # Unknown tool check
            for tc in action.tool_calls:
                if not self.registry.has(tc.name):
                    violations.append(
                        Violation(
                            code=ViolationCode.UNKNOWN_TOOL,
                            message=f"unknown tool: {tc.name}",
                            step_index=step_idx,
                            witness={"tool": tc.name},
                        )
                    )

            tool_results = injector.process_batch(action.tool_calls, self.registry)
            visible = [
                r
                for r in tool_results
                if self.config.record_dropped_results or not r.dropped
            ]
            step_cost = blowup_cost + sum(r.cost for r in tool_results if not r.dropped)
            cumulative_cost += sum(r.cost for r in tool_results if not r.dropped)

            step = TrajectoryStep(
                index=step_idx,
                observation=obs,
                action=action,
                tool_results=tool_results,
                faults_applied=injector.applied[step_faults_before:],
                cost=step_cost,
                cumulative_cost=cumulative_cost,
            )
            trajectory.steps.append(step)
            history.append(step)

            step_violations = evaluate_contracts(
                self.config.contracts, step, history[:-1], self.registry
            )
            violations.extend(step_violations)
            if step_violations and self.config.stop_on_violation:
                break

            # Present only non-dropped results to the next observation by default
            # (dropped ones stay in the trajectory for forensics).
            tool_results = [r for r in visible if not r.dropped]

        else:
            # Hit max_steps without final answer — MaxSteps contract may fire.
            pass

        trajectory.final_answer = final_answer
        trajectory.total_cost = cumulative_cost
        trajectory.total_tokens = total_tokens
        trajectory.metadata = {
            "faults_applied": list(injector.applied),
            "campaign": self.config.campaign.model_dump(mode="json"),
        }

        end_violations = evaluate_end(
            self.config.contracts, trajectory, self.registry
        )
        violations.extend(end_violations)

        if save_path:
            trajectory.save(save_path)

        verdict = Verdict.PASS if not violations else Verdict.FAIL
        return SimResult(
            verdict=verdict,
            seed=self.config.seed,
            violations=violations,
            steps=len(trajectory.steps),
            total_cost=cumulative_cost,
            total_tokens=total_tokens,
            trajectory_path=save_path,
            faults_applied=list(injector.applied),
            final_answer=final_answer,
            repro=SimResult.make_repro(
                seed=self.config.seed,
                trajectory=save_path,
            ),
        )

    def replay(
        self,
        trajectory: Trajectory,
        *,
        save_path: str | None = None,
        override_seed: int | None = None,
    ) -> SimResult:
        """Re-execute a recorded trajectory with the same (or override) seed."""
        seed = self.config.seed if override_seed is None else override_seed
        cfg = SimConfig(
            seed=seed,
            max_steps=self.config.max_steps,
            prompt=trajectory.prompt or self.config.prompt,
            campaign=self.config.campaign,
            contracts=list(self.config.contracts),
            stop_on_violation=self.config.stop_on_violation,
        )
        return Simulator(self.registry, cfg).run(
            FixtureAgent(trajectory), save_path=save_path
        )

    def record(
        self,
        agent: Agent,
        path: str,
    ) -> tuple[SimResult, Trajectory]:
        """Run and persist the trajectory JSON."""
        result = self.run(agent, save_path=path)
        traj = Trajectory.load(path)
        return result, traj


def actions_equal(a: AgentAction, b: AgentAction) -> bool:
    return a.model_dump() == b.model_dump()


def build_fixture_trajectory(
    *,
    seed: int,
    prompt: str,
    steps: list[dict[str, Any]],
) -> Trajectory:
    """Helper to hand-author CI fixtures without a live agent."""
    traj_steps: list[TrajectoryStep] = []
    for i, raw in enumerate(steps):
        action = AgentAction.model_validate(raw["action"])
        calls = [
            ToolCall.model_validate(c) if isinstance(c, dict) else c
            for c in action.tool_calls
        ]
        action.tool_calls = calls
        traj_steps.append(
            TrajectoryStep(
                index=i,
                observation=Observation(step=i, prompt=prompt),
                action=action,
            )
        )
    return Trajectory(seed=seed, prompt=prompt, steps=traj_steps)
