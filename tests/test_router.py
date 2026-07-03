"""Unit tests for Phase 4 cascade router."""
from __future__ import annotations
import os
import pytest
from tokengate.core.config import Settings
from tokengate.analytics.db import init_db


def make_settings(tmp_path) -> Settings:
    old = os.environ.get("TOKENGATE_DATA_DIR")
    os.environ["TOKENGATE_DATA_DIR"] = str(tmp_path)
    s = Settings()
    init_db(s.db_path)
    if old is None:
        os.environ.pop("TOKENGATE_DATA_DIR", None)
    else:
        os.environ["TOKENGATE_DATA_DIR"] = old
    return s


def test_router_settings_defaults(tmp_path):
    s = make_settings(tmp_path)
    assert s.router_enabled is True
    assert s.router_difficulty_threshold == 0.4
    assert s.router_escalation_enabled is True
    assert s.router_escalation_threshold == 3
    assert s.router_tools_tier == "strong"
    assert s.router_cheap_model == {"anthropic": "claude-haiku-4-5", "openai": "gpt-4o-mini"}
    assert s.router_strong_model == {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o"}
