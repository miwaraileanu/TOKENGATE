import json
import os
import signal
import socket
import sys
import time
import pytest
from pathlib import Path
from tokengate.cli.daemon import (
    read_pid_file, write_pid_file, remove_pid_file,
    is_port_free, check_port_or_exit, _pid_alive,
)


@pytest.fixture
def pid_path(tmp_path):
    return tmp_path / "tokengate.pid"


def test_write_and_read_pid_file(pid_path):
    write_pid_file(pid_path, pid=12345, port=8787)
    data = read_pid_file(pid_path)
    assert data["pid"] == 12345
    assert data["port"] == 8787
    assert "started_at" in data


def test_read_missing_pid_file_returns_none(tmp_path):
    assert read_pid_file(tmp_path / "missing.pid") is None


def test_remove_pid_file(pid_path):
    write_pid_file(pid_path, pid=1, port=8787)
    remove_pid_file(pid_path)
    assert not pid_path.exists()


def test_remove_missing_pid_file_is_noop(tmp_path):
    remove_pid_file(tmp_path / "missing.pid")  # must not raise


def test_pid_alive_current_process():
    assert _pid_alive(os.getpid()) is True


def test_pid_alive_dead_process():
    # PID 1 is always alive on Unix, so use an absurdly high PID
    assert _pid_alive(999999999) is False


def test_is_port_free_on_unused_port():
    # Find a free port
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert is_port_free("127.0.0.1", port) is True


def test_is_port_free_on_occupied_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    assert is_port_free("127.0.0.1", port) is False
    s.close()


def test_check_port_busy_exits(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    pid_path = tmp_path / "t.pid"
    with pytest.raises(SystemExit) as exc:
        check_port_or_exit("127.0.0.1", port, pid_path)
    assert exc.value.code == 1
    s.close()


def test_stale_pid_file_detected(pid_path):
    write_pid_file(pid_path, pid=999999999, port=8787)
    # _pid_alive(999999999) is False, so stale
    data = read_pid_file(pid_path)
    assert data is not None
    assert not _pid_alive(data["pid"])
