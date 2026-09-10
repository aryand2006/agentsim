"""Fault injection campaign tests — bit-reproducible seeds."""

from __future__ import annotations

from agentsim.faults import FaultCampaign, FaultInjector, FaultKind
from agentsim.models import ToolCall
from agentsim.runner import SimConfig, Simulator


def test_same_seed_same_faults(registry) -> None:
    campaign = FaultCampaign.all_kinds(
        timeout_prob=0.5,
        error_prob=0.5,
        drop_prob=0.3,
        reorder_prob=0.9,
        poison_prob=0.9,
        budget_blowup_prob=0.5,
    )
    calls = [
        ToolCall(name="get_balance", arguments={"account": "alice"}, call_id="a"),
        ToolCall(name="get_balance", arguments={"account": "bob"}, call_id="b"),
    ]

    a = FaultInjector(seed=12345, campaign=campaign)
    b = FaultInjector(seed=12345, campaign=campaign)
    # Consume poison + blowup the same way.
    sa = a.poison_schemas(registry)
    sb = b.poison_schemas(registry)
    assert sa == sb
    assert a.maybe_blowup_tokens(10, 10) == b.maybe_blowup_tokens(10, 10)
    ra = a.process_batch(calls, registry)
    rb = b.process_batch(calls, registry)
    assert [r.model_dump() for r in ra] == [r.model_dump() for r in rb]
    assert a.applied == b.applied


def test_different_seeds_diverge(registry) -> None:
    campaign = FaultCampaign(
        enabled=[FaultKind.TOOL_ERROR],
        error_prob=0.5,
    )
    calls = [
        ToolCall(name="get_balance", arguments={"account": "alice"}, call_id="a")
    ]
    outcomes = set()
    for seed in range(50):
        inj = FaultInjector(seed=seed, campaign=campaign)
        results = inj.process_batch(calls, registry)
        outcomes.add(results[0].ok)
    assert outcomes == {True, False}


def test_timeout_fault(registry) -> None:
    campaign = FaultCampaign(
        enabled=[FaultKind.TIMEOUT],
        timeout_prob=1.0,
    )
    inj = FaultInjector(seed=1, campaign=campaign)
    results = inj.process_batch(
        [ToolCall(name="get_balance", arguments={"account": "alice"}, call_id="x")],
        registry,
    )
    assert results[0].timed_out
    assert results[0].ok is False
    assert any(f.startswith("timeout:") for f in inj.applied)


def test_drop_result(registry) -> None:
    campaign = FaultCampaign(enabled=[FaultKind.DROP_RESULT], drop_prob=1.0)
    inj = FaultInjector(seed=1, campaign=campaign)
    results = inj.process_batch(
        [ToolCall(name="get_balance", arguments={"account": "alice"}, call_id="x")],
        registry,
    )
    assert results[0].dropped
    assert results[0].output is None


def test_reorder(registry) -> None:
    campaign = FaultCampaign(enabled=[FaultKind.REORDER], reorder_prob=1.0)
    # Find a seed that actually reorders.
    found = False
    for seed in range(200):
        inj = FaultInjector(seed=seed, campaign=campaign)
        calls = [
            ToolCall(name="get_balance", arguments={"account": "alice"}, call_id="a"),
            ToolCall(name="get_balance", arguments={"account": "bob"}, call_id="b"),
        ]
        inj.process_batch(calls, registry)
        if any(f.startswith("reorder:") for f in inj.applied):
            found = True
            break
    assert found


def test_poison_description(registry) -> None:
    campaign = FaultCampaign(
        enabled=[FaultKind.POISON_DESCRIPTION], poison_prob=1.0
    )
    inj = FaultInjector(seed=9, campaign=campaign)
    schemas = inj.poison_schemas(registry)
    assert any(s.get("_poisoned") for s in schemas)
    assert any("IGNORE" in s["description"] or "SYSTEM" in s["description"]
               or "injected" in s["description"] for s in schemas)


def test_budget_blowup_increases_cost(registry, safe_agent) -> None:
    clean = Simulator(
        registry,
        SimConfig(seed=0, campaign=FaultCampaign.none()),
    ).run(safe_agent.__class__())

    blowup = FaultCampaign(
        enabled=[FaultKind.BUDGET_BLOWUP],
        budget_blowup_prob=1.0,
        budget_multiplier=100.0,
    )
    noisy = Simulator(
        registry,
        SimConfig(seed=0, campaign=blowup),
    ).run(safe_agent.__class__())
    assert noisy.total_tokens > clean.total_tokens
    assert noisy.total_cost > clean.total_cost
    assert any("budget_blowup" in f for f in noisy.faults_applied)


def test_full_run_reproducible_under_chaos(registry, safe_agent) -> None:
    campaign = FaultCampaign.all_kinds(
        timeout_prob=0.2,
        error_prob=0.2,
        drop_prob=0.1,
        reorder_prob=0.5,
        poison_prob=0.5,
        budget_blowup_prob=0.3,
        max_faults_per_run=5,
    )
    config = SimConfig(seed=99, campaign=campaign)
    r1 = Simulator(registry, config).run(safe_agent.__class__())
    r2 = Simulator(registry, config).run(safe_agent.__class__())
    assert r1.faults_applied == r2.faults_applied
    assert r1.total_cost == r2.total_cost
    assert r1.total_tokens == r2.total_tokens
    assert r1.verdict == r2.verdict
