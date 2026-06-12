from __future__ import annotations
import typer

app = typer.Typer(help="rait — TokenGate CLI (Phase 1 scaffold)", add_completion=False)

# Commands will be added in Task 15. This stub ensures the CLI entry point is importable.


@app.command(hidden=True)
def _placeholder() -> None:
    """Placeholder — real commands added in Task 15."""
    typer.echo("No commands registered yet. See Task 15.")
