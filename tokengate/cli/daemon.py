from __future__ import annotations
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def write_pid_file(pid_path: Path, *, pid: int, port: int) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"pid": pid, "port": port, "started_at": time.time()}
    pid_path.write_text(json.dumps(data))


def read_pid_file(pid_path: Path) -> dict | None:
    if not pid_path.exists():
        return None
    try:
        return json.loads(pid_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def remove_pid_file(pid_path: Path) -> None:
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    if _HAS_PSUTIL:
        return _psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _pid_is_tokengate(pid: int) -> bool:
    """Best-effort check that a PID belongs to a tokengate process."""
    if not _HAS_PSUTIL:
        return True  # Can't verify, assume OK
    try:
        p = _psutil.Process(pid)
        cmdline = " ".join(p.cmdline())
        return "tokengate" in cmdline or "uvicorn" in cmdline
    except (_psutil.NoSuchProcess, _psutil.AccessDenied):
        return False


def is_port_free(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def check_port_or_exit(host: str, port: int, pid_path: Path) -> None:
    if not is_port_free(host, port):
        print(
            f"Port {port} in use — is TokenGate already running? Try `rait status`.",
            file=sys.stderr,
        )
        sys.exit(1)


def start_foreground(host: str, port: int, pid_path: Path, log_level: str = "info") -> None:
    import atexit
    write_pid_file(pid_path, pid=os.getpid(), port=port)
    atexit.register(remove_pid_file, pid_path)

    if sys.platform != "win32":
        def _sigint(sig, frame):
            remove_pid_file(pid_path)
            sys.exit(0)
        signal.signal(signal.SIGINT, _sigint)

    import uvicorn
    uvicorn.run(
        "tokengate.proxy.server:app",
        host=host, port=port, log_level=log_level,
    )


def start_detached(host: str, port: int, pid_path: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "tokengate.proxy.server:app",
        "--host", host, "--port", str(port),
    ]

    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=log_file,
            creationflags=flags, close_fds=True,
        )
    else:
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=log_file,
            start_new_session=True, close_fds=True,
        )

    # Give the process a moment to start, then write PID
    time.sleep(0.5)
    if proc.poll() is not None:
        print(f"Failed to start TokenGate. Check logs: {log_path}", file=sys.stderr)
        sys.exit(1)

    write_pid_file(pid_path, pid=proc.pid, port=port)
    print(f"TokenGate started (PID {proc.pid}) on port {port}. Logs: {log_path}")


def stop_daemon(pid_path: Path) -> None:
    data = read_pid_file(pid_path)
    if data is None:
        print("TokenGate is not running.")
        return

    pid = data["pid"]
    if not _pid_alive(pid):
        print("Not running (stale PID file removed).")
        remove_pid_file(pid_path)
        return

    if not _pid_is_tokengate(pid):
        print(f"WARNING: PID {pid} does not appear to be a TokenGate process. Aborting stop.")
        return

    try:
        if sys.platform == "win32":
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            os.kill(pid, signal.SIGTERM)
        remove_pid_file(pid_path)
        print(f"TokenGate stopped (PID {pid}).")
    except OSError as e:
        print(f"Failed to stop process {pid}: {e}", file=sys.stderr)


def status_daemon(pid_path: Path) -> None:
    data = read_pid_file(pid_path)
    if data is None:
        print("TokenGate is not running.")
        return

    pid = data["pid"]
    if not _pid_alive(pid):
        print("Not running (stale PID file removed).")
        remove_pid_file(pid_path)
        return

    uptime_s = int(time.time() - data["started_at"])
    h, rem = divmod(uptime_s, 3600)
    m, s = divmod(rem, 60)
    uptime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    print(f"TokenGate is running.")
    print(f"  PID   : {pid}")
    print(f"  Port  : {data['port']}")
    print(f"  Uptime: {uptime_str}")
    print(f"  Dashboard: http://127.0.0.1:{data['port']}/dashboard")
