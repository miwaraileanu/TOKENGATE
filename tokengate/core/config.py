from __future__ import annotations
import os
import yaml
from pathlib import Path


_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-3.5-turbo": (0.50, 1.50),
}


class Settings:
    def __init__(self, config_path: Path | str | None = None):
        data_dir_env = os.environ.get("TOKENGATE_DATA_DIR")
        self.data_dir = (
            Path(data_dir_env).expanduser()
            if data_dir_env
            else Path("~/.rait").expanduser()
        )

        raw: dict = {}
        cfg = Path(config_path) if config_path else (self.data_dir / "tokengate.yaml")
        if cfg.exists():
            with open(cfg) as f:
                raw = yaml.safe_load(f) or {}

        self.bind: str = os.environ.get("TOKENGATE_BIND", raw.get("bind", "127.0.0.1"))
        self.port: int = int(os.environ.get("TOKENGATE_PORT", str(raw.get("port", 8787))))
        self.tokengate_key: str = os.environ.get("TOKENGATE_KEY", raw.get("tokengate_key", ""))
        self.log_level: str = raw.get("log_level", "info")
        self.openai_base_url: str = raw.get("openai_base_url", "https://api.openai.com")
        self.anthropic_base_url: str = raw.get("anthropic_base_url", "https://api.anthropic.com")

        yaml_prices = raw.get("prices", {})
        self.prices: dict[str, tuple[float, float]] = {
            **_DEFAULT_PRICES,
            **{k: tuple(v) for k, v in yaml_prices.items()},
        }

        self.db_path: Path = self.data_dir / "tokengate.db"
        self.pid_path: Path = self.data_dir / "tokengate.pid"
        self.log_path: Path = self.data_dir / "logs" / "tokengate.log"
