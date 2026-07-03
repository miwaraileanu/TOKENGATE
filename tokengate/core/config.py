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

        _c = raw.get("cache", {})
        self.cache_exact_ttl: int = int(_c.get("exact_ttl_seconds", 86400))
        self.cache_semantic_threshold: float = float(_c.get("semantic_threshold", 0.93))
        self.cache_max_entries: int = int(_c.get("max_entries", 50000))
        self.cache_blocklist: list = list(_c.get("blocklist_patterns", [
            r"\btoday\b", r"\bnow\b", r"\blatest\b", r"\bprice\b",
        ]))
        self.cache_serve_unverified: bool = bool(_c.get("serve_unverified", False))

        _d = raw.get("distill", {})
        self.distill_threshold_tokens: int = int(_d.get("threshold_tokens", 6000))
        self.distill_keep_recent_turns: int = int(_d.get("keep_recent_turns", 4))
        self.distill_top_k: int = int(_d.get("top_k", 3))
        self.distill_ttl_seconds: int = int(_d.get("ttl_seconds", 86400))
        _dm = _d.get("model", {})
        self.distill_model: dict[str, str] = {
            "anthropic": _dm.get("anthropic", "claude-haiku-4-5"),
            "openai": _dm.get("openai", "gpt-4o-mini"),
        }

        _b = raw.get("budget", {})
        self.budget_chat: int = int(_b.get("chat", 1024))
        self.budget_extraction: int = int(_b.get("extraction", 512))
        self.budget_code: int = int(_b.get("code", 2048))
        self.budget_long_form: int = int(_b.get("long_form", 4096))
        self.budget_extraction_instruction: str = _b.get(
            "extraction_instruction",
            "Answer with the requested data only, no preamble.",
        )
        self.budget_extraction_instruction_enabled: bool = bool(
            _b.get("extraction_instruction_enabled", True)
        )
        self.budget_table: dict[str, int] = {
            "chat": self.budget_chat,
            "extraction": self.budget_extraction,
            "code": self.budget_code,
            "long_form": self.budget_long_form,
        }

        _r = raw.get("router", {})
        self.router_enabled: bool = bool(_r.get("enabled", True))
        self.router_difficulty_threshold: float = float(_r.get("difficulty_threshold", 0.4))
        self.router_escalation_enabled: bool = bool(_r.get("escalation_enabled", True))
        self.router_escalation_threshold: int = int(_r.get("escalation_threshold", 3))
        self.router_tools_tier: str = _r.get("tools_tier", "strong")
        _cm = _r.get("cheap_model", {})
        self.router_cheap_model: dict[str, str] = {
            "anthropic": _cm.get("anthropic", "claude-haiku-4-5"),
            "openai": _cm.get("openai", "gpt-4o-mini"),
        }
        _sm = _r.get("strong_model", {})
        self.router_strong_model: dict[str, str] = {
            "anthropic": _sm.get("anthropic", "claude-sonnet-4-6"),
            "openai": _sm.get("openai", "gpt-4o"),
        }
