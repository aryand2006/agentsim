"""Shared fixtures for agentsim tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from banking_agent import (  # noqa: E402
    BankingAgent,
    NaughtyBankingAgent,
    PoisonableAgent,
    build_registry,
    reset_state,
)


@pytest.fixture(autouse=True)
def _reset_ledger() -> None:
    reset_state()


@pytest.fixture
def registry():
    return build_registry()


@pytest.fixture
def safe_agent():
    return BankingAgent()


@pytest.fixture
def naughty_agent():
    return NaughtyBankingAgent()


@pytest.fixture
def poisonable_agent():
    return PoisonableAgent()
