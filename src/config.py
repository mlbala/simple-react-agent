"""Configuration loaded from environment variables (optionally via a `.env` file).

API keys stay in the environment: `ChatOpenAI` reads `OPENAI_API_KEY` and the
search tool reads `TAVILY_API_KEY` directly, so secrets never pass through the UI.
"""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Cost-efficient OpenAI model with tool calling (verified in OpenAI's model docs, Sept 2026).
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "TAVILY_API_KEY")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    openai_model: str
    log_level: str


def load_settings() -> Settings:
    """Load and validate settings, raising `ConfigError` with an actionable message."""
    load_dotenv(override=False)  # Real environment variables win over `.env`.

    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name, "").strip()]
    if missing:
        raise ConfigError(
            f"Missing required setting(s): {', '.join(missing)}. "
            "Copy `.env.example` to `.env`, fill in the values, and restart the app."
        )

    log_level = os.getenv("LOG_LEVEL", "").strip().upper() or "INFO"
    if log_level not in LOG_LEVELS:
        raise ConfigError(f"LOG_LEVEL must be one of {', '.join(LOG_LEVELS)} (got {log_level!r}).")

    model = os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
    return Settings(openai_model=model, log_level=log_level)


def configure_logging(level: str = "INFO") -> None:
    """Configure basic application logging. Safe to call on every Streamlit rerun."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(level)
    # HTTP client libraries log every request at INFO/DEBUG; keep them quiet by default.
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
