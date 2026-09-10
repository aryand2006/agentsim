"""Contract violation tests."""

from __future__ import annotations

from agentsim.contracts import (
    AllowedTools,
    ApprovalBeforeSideEffect,
    ForbiddenTools,
    MaxCost,
    MaxSteps,
    RequiredToolSequence,
)
from agentsim.faults import FaultCampaign, FaultKind
from agentsim.result import Verdict, ViolationCode
from agentsim.runner import SimConfig, Simulator


def test_approval_before_side_effect_fails(registry, naughty_agent) -> None:
    config = SimConfig(
        seed=0,
        campaign=FaultCampaign.none(),
        contracts=[ApprovalBeforeSideEffect({"request_approval"})],
    )
    result = Simulator(registry, config).run(naughty_agent)
    assert result.verdict == Verdict.FAIL
    assert result.violations[0].code == ViolationCode.APPROVAL_REQUIRED
    assert "repro:" in result.summarize()
    assert f"--seed {result.seed}" in result.repro


def test_forbidden_tool(registry, poisonable_agent) -> None:
    # Force poison so the agent may call delete_all.
    campaign = FaultCampaign(
        enabled=[FaultKind.POISON_DESCRIPTION],
        poison_prob=1.0,
    )
    # Find a seed where poison mentions delete_all.
    failed = None
    for seed in range(100):
        config = SimConfig(
            seed=seed,
            campaign=campaign,
            contracts=[ForbiddenTools({"delete_all"})],
        )
        result = Simulator(registry, config).run(poisonable_agent.__class__())
        if result.verdict == Verdict.FAIL and any(
            v.code == ViolationCode.FORBIDDEN_TOOL for v in result.violations
        ):
            failed = result
            break
    # Even if poison didn't trigger delete_all, agent might still PASS —
    # assert that when it fails, witness is correct. Also assert forbidden works
    # via a direct naughty path:
    if failed is None:
        from agentsim.models import AgentAction, Observation, ToolCall

        class WipeAgent:
            def step(self, observation: Observation) -> AgentAction:
                return AgentAction(
                    kind="tool_calls",
                    tool_calls=[ToolCall(name="delete_all", arguments={})],
                )

        result = Simulator(
            registry,
            SimConfig(
                seed=0,
                campaign=FaultCampaign.none(),
                contracts=[ForbiddenTools({"delete_all"})],
            ),
        ).run(WipeAgent())
        assert result.verdict == Verdict.FAIL
        assert result.violations[0].code == ViolationCode.FORBIDDEN_TOOL
    else:
        assert failed.violations[0].step_index >= 0


def test_required_sequence_missing(registry, naughty_agent) -> None:
    config = SimConfig(
        seed=0,
        campaign=FaultCampaign.none(),
        contracts=[
            RequiredToolSequence(
                ["get_balance", "request_approval", "transfer_funds"]
            )
        ],
    )
    result = Simulator(registry, config).run(naughty_agent)
    assert result.verdict == Verdict.FAIL
    assert any(v.code == ViolationCode.REQUIRED_SEQUENCE for v in result.violations)


def test_allowed_tools(registry, safe_agent) -> None:
    config = SimConfig(
        seed=0,
        campaign=FaultCampaign.none(),
        contracts=[AllowedTools({"get_balance"})],  # transfer not allowed
    )
    result = Simulator(registry, config).run(safe_agent)
    assert result.verdict == Verdict.FAIL
    assert result.violations[0].code == ViolationCode.ALLOWED_TOOLS


def test_max_steps(registry, safe_agent) -> None:
    SimConfig(
        seed=0,
        max_steps=1,
        campaign=FaultCampaign.none(),
        contracts=[MaxSteps(1)],
        stop_on_violation=False,
    )
    # Agent needs >1 step; with max_steps=1 the loop ends after first tool call.
    # MaxSteps(1) allows index 0 only — if somehow more, fail. With max_steps=1
    # we only get one step which is OK for MaxSteps(1). Use MaxSteps(0) edge:
    config2 = SimConfig(
        seed=0,
        max_steps=5,
        campaign=FaultCampaign.none(),
        contracts=[MaxSteps(2)],
    )
    result = Simulator(registry, config2).run(safe_agent)
    # Safe agent: step0 balance, step1 approval, step2 transfer → FAIL at step index 2
    assert result.verdict == Verdict.FAIL
    assert any(v.code == ViolationCode.MAX_STEPS for v in result.violations)


def test_max_cost_with_blowup(registry, safe_agent) -> None:
    campaign = FaultCampaign(
        enabled=[FaultKind.BUDGET_BLOWUP],
        budget_blowup_prob=1.0,
        budget_multiplier=10_000.0,
    )
    config = SimConfig(
        seed=0,
        campaign=campaign,
        contracts=[MaxCost(0.001)],
    )
    result = Simulator(registry, config).run(safe_agent)
    assert result.verdict == Verdict.FAIL
    assert any(v.code == ViolationCode.MAX_COST for v in result.violations)
    assert result.violations[0].step_index >= 0
