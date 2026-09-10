"""CLI entrypoints for agentsim."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import click

from agentsim import __version__
from agentsim.contracts.base import (
    AllowedTools,
    ApprovalBeforeSideEffect,
    ForbiddenTools,
    MaxCost,
    MaxSteps,
    RequiredToolSequence,
)
from agentsim.faults.injector import FaultCampaign, FaultKind
from agentsim.models import Trajectory
from agentsim.result import SimResult
from agentsim.runner import SimConfig, Simulator
from agentsim.shrink import shrink_trajectory


def _load_campaign(path: str | None, kinds: tuple[str, ...]) -> FaultCampaign:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "enabled" in data:
            data["enabled"] = [FaultKind(k) for k in data["enabled"]]
        return FaultCampaign.model_validate(data)
    if kinds:
        return FaultCampaign(enabled=[FaultKind(k) for k in kinds])
    return FaultCampaign.none()


def _load_contracts(path: str | None) -> list[Any]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    contracts: list[Any] = []
    for item in data:
        kind = item["type"]
        if kind == "required_sequence":
            contracts.append(RequiredToolSequence(item["sequence"]))
        elif kind == "allowed_tools":
            contracts.append(AllowedTools(item["tools"]))
        elif kind == "forbidden_tools":
            contracts.append(ForbiddenTools(item["tools"]))
        elif kind == "max_cost":
            contracts.append(MaxCost(float(item["limit"])))
        elif kind == "max_steps":
            contracts.append(MaxSteps(int(item["limit"])))
        elif kind == "approval_before_side_effect":
            contracts.append(
                ApprovalBeforeSideEffect(item.get("approval_tools"))
            )
        else:
            raise click.ClickException(f"unknown contract type: {kind}")
    return contracts


def _import_example() -> tuple[Any, Any]:
    """Load the bundled example agent + registry (examples must be on path)."""
    root = Path(__file__).resolve().parents[3]
    examples = root / "examples"
    if examples.is_dir() and str(examples) not in sys.path:
        sys.path.insert(0, str(examples))
    # Also try CWD/examples for editable / source checkouts.
    cwd_ex = Path.cwd() / "examples"
    if cwd_ex.is_dir() and str(cwd_ex) not in sys.path:
        sys.path.insert(0, str(cwd_ex))
    mod = importlib.import_module("banking_agent")
    return mod.build_registry(), mod.BankingAgent


@click.group()
@click.version_option(__version__, prog_name="agentsim")
def cli() -> None:
    """Deterministic seeded fault simulation for tool-using AI agents."""


@cli.command("run")
@click.option("--seed", default=0, show_default=True, help="RNG seed for faults.")
@click.option("--campaign", "campaign_path", default=None, help="Fault campaign JSON.")
@click.option(
    "--fault",
    "faults",
    multiple=True,
    type=click.Choice([k.value for k in FaultKind], case_sensitive=False),
    help="Enable a fault kind (repeatable).",
)
@click.option("--contracts", "contracts_path", default=None, help="Contracts JSON.")
@click.option("--replay", "replay_path", default=None, help="Replay a trajectory JSON.")
@click.option("--record", "record_path", default=None, help="Write trajectory JSON.")
@click.option("--max-steps", default=32, show_default=True)
@click.option("--prompt", default="Transfer $50 to Alice after approval.")
def run_cmd(
    seed: int,
    campaign_path: str | None,
    faults: tuple[str, ...],
    contracts_path: str | None,
    replay_path: str | None,
    record_path: str | None,
    max_steps: int,
    prompt: str,
) -> None:
    """Run or replay an agent simulation; print PASS/FAIL with repro."""
    registry, agent_cls = _import_example()
    campaign = _load_campaign(campaign_path, faults)
    contracts = _load_contracts(contracts_path)
    # Always enforce a max-steps ceiling from the flag if not already present.
    if not any(getattr(c, "name", "") == "max_steps" for c in contracts):
        contracts.append(MaxSteps(max_steps))

    config = SimConfig(
        seed=seed,
        max_steps=max_steps,
        prompt=prompt,
        campaign=campaign,
        contracts=contracts,
    )
    sim = Simulator(registry, config)

    if replay_path:
        traj = Trajectory.load(replay_path)
        result = sim.replay(traj, save_path=record_path)
    else:
        agent = agent_cls()
        result = sim.run(agent, save_path=record_path)

    result.repro = SimResult.make_repro(
        seed=seed,
        campaign=campaign_path,
        trajectory=replay_path or record_path,
        contracts=contracts_path,
    )
    click.echo(result.summarize())
    raise SystemExit(0 if result.ok else 1)


@cli.command("shrink")
@click.argument("trajectory_path", type=click.Path(exists=True))
@click.option("--seed", default=None, type=int, help="Override seed (default: traj).")
@click.option("--campaign", "campaign_path", default=None)
@click.option(
    "--fault",
    "faults",
    multiple=True,
    type=click.Choice([k.value for k in FaultKind], case_sensitive=False),
)
@click.option("--contracts", "contracts_path", default=None)
@click.option("--out", "out_path", default=None, help="Write minimized trajectory.")
def shrink_cmd(
    trajectory_path: str,
    seed: int | None,
    campaign_path: str | None,
    faults: tuple[str, ...],
    contracts_path: str | None,
    out_path: str | None,
) -> None:
    """Minimize a failing trajectory while preserving FAIL (best-effort)."""
    registry, _ = _import_example()
    traj = Trajectory.load(trajectory_path)
    campaign = _load_campaign(campaign_path, faults)
    contracts = _load_contracts(contracts_path)
    config = SimConfig(
        seed=seed if seed is not None else traj.seed,
        campaign=campaign,
        contracts=contracts,
        max_steps=max(32, len(traj.steps) + 4),
    )
    minimized, result = shrink_trajectory(traj, registry, config)
    dest = out_path or str(Path(trajectory_path).with_suffix(".shrunk.json"))
    minimized.save(dest)
    click.echo(f"shrunk {len(traj.steps)} → {len(minimized.steps)} steps → {dest}")
    click.echo(result.summarize())
    raise SystemExit(0 if result.ok else 1)


@cli.command("campaign")
@click.option("--seed-start", default=0, show_default=True)
@click.option("--seed-count", default=20, show_default=True)
@click.option("--campaign", "campaign_path", default=None)
@click.option(
    "--fault",
    "faults",
    multiple=True,
    type=click.Choice([k.value for k in FaultKind], case_sensitive=False),
)
@click.option("--contracts", "contracts_path", default=None)
@click.option("--max-steps", default=32, show_default=True)
@click.option("--replay", "replay_path", default=None)
def campaign_cmd(
    seed_start: int,
    seed_count: int,
    campaign_path: str | None,
    faults: tuple[str, ...],
    contracts_path: str | None,
    max_steps: int,
    replay_path: str | None,
) -> None:
    """Sweep seeds and report PASS/FAIL counts (chaos campaign)."""
    registry, agent_cls = _import_example()
    campaign = _load_campaign(campaign_path, faults)
    if not campaign.enabled:
        campaign = FaultCampaign.all_kinds(
            timeout_prob=0.25,
            error_prob=0.25,
            drop_prob=0.1,
            reorder_prob=0.5,
            poison_prob=0.4,
            budget_blowup_prob=0.2,
        )
    contracts = _load_contracts(contracts_path)
    if not contracts:
        contracts = [
            ApprovalBeforeSideEffect({"request_approval"}),
            ForbiddenTools({"delete_all"}),
            MaxCost(10.0),
            MaxSteps(max_steps),
        ]

    traj = Trajectory.load(replay_path) if replay_path else None
    passed = 0
    failed = 0
    for i in range(seed_count):
        seed = seed_start + i
        config = SimConfig(
            seed=seed,
            max_steps=max_steps,
            campaign=campaign,
            contracts=contracts,
        )
        sim = Simulator(registry, config)
        if traj is not None:
            result = sim.replay(traj)
        else:
            result = sim.run(agent_cls())
        if result.ok:
            passed += 1
            click.echo(f"PASS seed={seed}")
        else:
            failed += 1
            click.echo(result.summarize())

    click.echo(f"\ncampaign: PASS={passed} FAIL={failed} / {seed_count}")
    raise SystemExit(0 if failed == 0 else 1)


@cli.command("demo")
@click.argument("which", type=click.Choice(["happy", "naughty", "poison"]), default="happy")
@click.option("--seed", default=0, show_default=True)
def demo_cmd(which: str, seed: int) -> None:
    """Run bundled banking demos (happy / naughty / poison)."""
    registry, _ = _import_example()
    import banking_agent as ex  # type: ignore  # noqa: PLC0415 — after path inject

    if which == "happy":
        agent: Any = ex.BankingAgent()
        contracts: list[Any] = [
            RequiredToolSequence(["get_balance", "request_approval", "transfer_funds"]),
            ApprovalBeforeSideEffect(),
            ForbiddenTools({"delete_all"}),
        ]
        campaign = FaultCampaign.none()
    elif which == "naughty":
        agent = ex.NaughtyBankingAgent()
        contracts = [ApprovalBeforeSideEffect()]
        campaign = FaultCampaign.none()
    else:
        agent = ex.PoisonableAgent()
        contracts = [
            ForbiddenTools({"delete_all"}),
            ApprovalBeforeSideEffect(),
        ]
        campaign = FaultCampaign(
            enabled=[FaultKind.POISON_DESCRIPTION],
            poison_prob=1.0,
        )

    config = SimConfig(seed=seed, campaign=campaign, contracts=contracts)
    result = Simulator(registry, config).run(agent)
    click.echo(result.summarize())
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    cli()
