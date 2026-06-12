from __future__ import annotations
import os
import pytest
from pathlib import Path
from tokengate.core.mock_provider import MockTransport


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Sets TOKENGATE_DATA_DIR to a temp path for the duration of the test."""
    monkeypatch.setenv("TOKENGATE_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def mock_transport():
    return MockTransport(mode="normal")


@pytest.fixture
def error_transport():
    return MockTransport(mode="error", error_status=500)


@pytest.fixture
def error_429_transport():
    return MockTransport(mode="error", error_status=429)
