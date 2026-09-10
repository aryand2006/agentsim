"""Record / replay trajectory tests."""

from __future__ import annotations

from pathlib import Path

from agentsim.contracts import (
    ApprovalBeforeSideEffect,
    ForbiddenTools,
    MaxCost,
    MaxSteps,
    RequiredToolSequence,
)
from agentsim.faults import FaultCampaign
from agentsim.models import Trajectory
from agentsim.result import Verdict
from agentsim.runner import FixtureAgent, SimConfig, Simulator


def test_record_and_reload(tmp_path: Path, registry, safe_agent) -> None:
    path = tmp_path / "traj.json"
    config = SimConfig(
        seed=7,
        campaign=FaultCampaign.none(),
        contracts=[
            RequiredToolSequence(
                ["get_balance", "request_approval", "transfer_funds"]
            ),
            ApprovalBeforeSideEffect({"request_approval"}),
            ForbiddenTools({"delete_all"}),
            MaxSteps(16),
        ],
    )
    sim = Simulator(registry, config)
    result = sim.run(safe_agent, save_path=str(path))
    assert result.verdict == Verdict.PASS
    assert path.exists()

    loaded = Trajectory.load(str(path))
    assert loaded.seed == 7
    assert len(loaded.steps) >= 3
    assert loaded.final_answer is not None


def test_replay_is_deterministic(tmp_path: Path, registry, safe_agent) -> None:
    path = tmp_path / "traj.json"
    config = SimConfig(seed=42, campaign=FaultCampaign.none(), contracts=[MaxSteps(16)])
    sim = Simulator(registry, config)
    r1 = sim.run(safe_agent, save_path=str(path))
    traj = Trajectory.load(str(path))

    # Replay twice with same seed → identical verdict/steps/cost.
    r2 = sim.replay(traj)
    r3 = Simulator(registry, config).replay(traj)
    assert r1.verdict == r2.verdict == r3.verdict == Verdict.PASS
    assert r2.steps == r3.steps
    assert r2.total_cost == r3.total_cost
    assert r2.faults_applied == r3.faults_applied


def test_fixture_agent_no_llm(registry) -> None:
    """CI path: hand-built trajectory + FixtureAgent, zero network."""
    from agentsim.models import AgentAction, Observation, ToolCall, TrajectoryStep

    steps = [
        TrajectoryStep(
            index=0,
            observation=Observation(step=0),
            action=AgentAction(
                kind="tool_calls",
                tool_calls=[
                    ToolCall(name="get_balance", arguments={"account": "alice"})
                ],
            ),
        ),
        TrajectoryStep(
            index=1,
            observation=Observation(step=1),
            action=AgentAction(
                kind="tool_calls",
                tool_calls=[
                    ToolCall(
                        name="request_approval",
                        arguments={"reason": "ok"},
                    )
                ],
            ),
        ),
        TrajectoryStep(
            index=2,
            observation=Observation(step=2),
            action=AgentAction(
                kind="tool_calls",
                tool_calls=[
                    ToolCall(
                        name="transfer_funds",
                        arguments={"to": "alice", "amount": 10.0},
                    )
                ],
            ),
        ),
        TrajectoryStep(
            index=3,
            observation=Observation(step=3),
            action=AgentAction(kind="final", answer="ok"),
        ),
    ]
    traj = Trajectory(seed=0, steps=steps)
    config = SimConfig(
        seed=0,
        campaign=FaultCampaign.none(),
        contracts=[
            RequiredToolSequence(
                ["get_balance", "request_approval", "transfer_funds"]
            ),
            MaxCost(1.0),
        ],
    )
    result = Simulator(registry, config).run(FixtureAgent(traj))
    assert result.ok
    assert result.final_answer == "ok"
