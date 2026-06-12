from __future__ import annotations
import os
import stat
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt, Confirm

from tokengate.analytics.db import init_db
from tokengate.core.config import Settings

console = Console()

_BANNER = """
[bold cyan]╔════════════════════════════════════╗
║       TokenGate  v0.1.0            ║
║  Intelligent Token-Saving Gateway  ║
╚════════════════════════════════════╝[/bold cyan]
"""

_SNIPPET_ANTHROPIC = """
[bold green]Integration (one-line change):[/bold green]
  [cyan]client = Anthropic(base_url="http://localhost:{port}")[/cyan]
"""

_SNIPPET_OPENAI = """
[bold green]Integration (one-line change):[/bold green]
  [cyan]client = OpenAI(base_url="http://localhost:{port}/v1")[/cyan]
"""


def run_wizard(
    provider: str | None = None,
    port: int | None = None,
    yes: bool = False,
) -> None:
    console.print(_BANNER)

    # ── 1. Provider ──────────────────────────────────────────────────────────
    if provider is None:
        provider = Prompt.ask(
            "Provider",
            choices=["anthropic", "openai", "both"],
            default="anthropic",
        )

    # ── 2. API key ────────────────────────────────────────────────────────────
    data_dir_env = os.environ.get("TOKENGATE_DATA_DIR")
    data_dir = (
        Path(data_dir_env).expanduser()
        if data_dir_env
        else Path("~/.rait").expanduser()
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    env_path = data_dir / ".env"

    key_var = "ANTHROPIC_API_KEY" if provider in ("anthropic", "both") else "OPENAI_API_KEY"
    existing = os.environ.get(key_var, "")

    if existing and yes:
        api_key = existing
    else:
        api_key = Prompt.ask(f"{key_var} (hidden)", password=True)

    _write_env(env_path, key_var, api_key)
    console.print(
        f"[green]✓[/green] API key written to [bold]{env_path}[/bold] (owner-only permissions)\n"
        "[yellow]Note:[/yellow] Key is stored but [bold]not validated[/bold]. "
        "Run [cyan]rait test --live[/cyan] (coming in a later release) to verify."
    )

    # ── 3. Port ───────────────────────────────────────────────────────────────
    if port is None:
        if yes:
            port = 8787
        else:
            raw = Prompt.ask("Gateway port", default="8787")
            port = int(raw)

    # ── 4. Write config ────────────────────────────────────────────────────────
    cfg_path = data_dir / "tokengate.yaml"
    _write_config(cfg_path, port)
    console.print(f"[green]✓[/green] Config written to [bold]{cfg_path}[/bold]")

    # ── 5. Init DB ─────────────────────────────────────────────────────────────
    os.environ["TOKENGATE_DATA_DIR"] = str(data_dir)
    s = Settings()
    init_db(s.db_path)
    console.print(f"[green]✓[/green] Database initialised at [bold]{s.db_path}[/bold]")

    # ── 6. Integration snippet ────────────────────────────────────────────────
    snippet = _SNIPPET_ANTHROPIC if provider in ("anthropic", "both") else _SNIPPET_OPENAI
    console.print(snippet.format(port=port))

    # ── 7. Offer to start ─────────────────────────────────────────────────────
    if yes or Confirm.ask("Start TokenGate now (detached)?", default=True):
        from tokengate.cli.daemon import check_port_or_exit, start_detached
        check_port_or_exit("127.0.0.1", port, s.pid_path)
        start_detached("127.0.0.1", port, s.pid_path, s.log_path)


def _write_env(env_path: Path, key_var: str, value: str) -> None:
    lines = []
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if not line.startswith(f"{key_var}="):
                lines.append(line)
    lines.append(f"{key_var}={value}")
    env_path.write_text("\n".join(lines) + "\n")
    if sys.platform != "win32":
        env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    else:
        _win_set_owner_only(env_path)


def _write_config(cfg_path: Path, port: int) -> None:
    import shutil, tokengate
    default_cfg = Path(tokengate.__file__).parent.parent / "tokengate.yaml"
    if default_cfg.exists():
        shutil.copy(default_cfg, cfg_path)
    # Patch the port
    text = cfg_path.read_text() if cfg_path.exists() else "bind: '127.0.0.1'\n"
    lines = [l for l in text.splitlines() if not l.startswith("port:")]
    lines.insert(0, f"port: {port}")
    cfg_path.write_text("\n".join(lines) + "\n")


def _win_set_owner_only(path: Path) -> None:
    try:
        import subprocess
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{os.getlogin()}:(R,W)"],
            check=True, capture_output=True,
        )
    except Exception:
        pass  # Best-effort on Windows
