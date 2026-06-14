from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="rait — TokenGate CLI", add_completion=False)
console = Console()


def _get_settings() -> "Settings":
    from tokengate.core.config import Settings
    return Settings()


@app.command()
def install(
    provider: Optional[str] = typer.Option(None, help="anthropic | openai | both"),
    port: Optional[int] = typer.Option(None, help="Gateway port (default 8787)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept all defaults non-interactively"),
):
    """Interactive setup wizard."""
    from tokengate.cli.wizard import run_wizard
    run_wizard(provider=provider, port=port, yes=yes)


@app.command()
def start(
    detach: bool = typer.Option(False, "--detach", "-d", help="Run in background"),
):
    """Start the TokenGate gateway."""
    from tokengate.cli.daemon import check_port_or_exit, start_foreground, start_detached
    s = _get_settings()
    check_port_or_exit(s.bind, s.port, s.pid_path)
    if detach:
        start_detached(s.bind, s.port, s.pid_path, s.log_path)
    else:
        console.print(f"Starting TokenGate on [cyan]{s.bind}:{s.port}[/cyan] (foreground — Ctrl+C to stop)")
        start_foreground(s.bind, s.port, s.pid_path, s.log_level)


@app.command()
def stop():
    """Stop the TokenGate gateway."""
    from tokengate.cli.daemon import stop_daemon
    s = _get_settings()
    stop_daemon(s.pid_path)


@app.command()
def status():
    """Show gateway status."""
    from tokengate.cli.daemon import status_daemon
    s = _get_settings()
    status_daemon(s.pid_path)


@app.command()
def stats():
    """Print token savings summary."""
    from tokengate.analytics.stats import get_stats
    s = _get_settings()
    if not s.db_path.exists():
        console.print("[yellow]No analytics data yet. Start the gateway and make some requests.[/yellow]")
        raise typer.Exit(0)
    data = get_stats(s.db_path)
    console.print_json(json.dumps(data))


@app.command("cache")
def cache_cmd(
    clear: bool = typer.Option(False, "--clear", help="Clear all caches"),
):
    """Manage the cache (stub — all options implemented in Phase 2)."""
    console.print("[yellow]Cache management (including --clear) is available from Phase 2 onwards.[/yellow]")


@app.command("test")
def test_cmd():
    """Send a sample request through the gateway."""
    s = _get_settings()
    console.print(f"[cyan]Sending test request to http://{s.bind}:{s.port} ...[/cyan]")
    try:
        import httpx
        resp = httpx.post(
            f"http://{s.bind}:{s.port}/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Say hello."}]},
            timeout=10,
        )
        console.print(f"Status: {resp.status_code}")
        console.print(f"x-tokengate-cache: {resp.headers.get('x-tokengate-cache', 'n/a')}")
        console.print(f"x-tokengate-model: {resp.headers.get('x-tokengate-model', 'n/a')}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
