"""Shared fixtures. Adds the project root to sys.path so the flat modules import."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analytics  # noqa: E402
from ai_analyst import AIAnalyst  # noqa: E402
from data_generator import load_dataset  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def evidence(dataset):
    return analytics.build_evidence_pack(dataset)


@pytest.fixture()
def offline_analyst():
    """An analyst that never makes a network call, whatever the environment."""
    return AIAnalyst(api_key="", force_fallback=True)
