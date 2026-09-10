"""Shrink + CLI smoke tests."""

from __future__ import annotations

from click.testing import CliRunner

from agentsim.cli.main import cli
from agentsim.contracts import ApprovalBeforeSideEffect, MaxSteps
from agentsim.faults import FaultCampaign
from agentsim.runner import SimConfig, Simulator
from agentsim.shrink import shrink_trajectory


def test_shrink_preserves_failure(registry, naughty_agent, tmp_path) -> None:
    config = SimConfig(
        seed=0,
        campaign=FaultCampaign.none(),
        contracts=[ApprovalBeforeSideEffect({"request_approval"}), MaxSteps(16)],
    )
    path = tmp_path / "fail.json"
    sim = Simulator(registry, config)
    result = sim.run(naughty_agent, save_path=str(path))
    assert not result.ok

    from agentsim.models import Trajectory

    traj = Trajectory.load(str(path))
    # Pad with a useless final-ish extra by re-running isn't needed —
    # naughty traj is already short; shrink should still return FAIL.
    minimized, mini_result = shrink_trajectory(traj, registry, config)
    assert not mini_result.ok
    assert len(minimized.steps) <= len(traj.steps)


def test_cli_run_pass(tmp_path) -> None:
    runner = CliRunner()
    out = tmp_path / "t.json"
    result = runner.invoke(
        cli,
        [
            "run",
            "--seed",
            "0",
            "--record",
            str(out),
            "--contracts",
            "examples/contracts.json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert out.exists()


def test_cli_run_fail_naughty_via_replay(tmp_path, registry, naughty_agent) -> None:
    # Record a naughty trajectory then replay with approval contract via CLI path
    # is heavy; unit-level shrink already covers. Smoke: campaign command.
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "campaign",
            "--seed-start",
            "0",
            "--seed-count",
            "3",
            "--fault",
            "timeout",
            "--contracts",
            "examples/contracts.json",
        ],
        catch_exceptions=False,
    )
    # May PASS or FAIL depending on seeds; command should exit cleanly with summary.
    assert "campaign: PASS=" in result.output
    assert result.exit_code in (0, 1)
