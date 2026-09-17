"""Centralized configuration and environment loader.

Follows python-pro conventions: dataclass configuration, pathlib paths, full type hints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Base directory
BASE_DIR: Path = Path(__file__).resolve().parent

# Load environment variables
ENV_PATH: Path = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration loaded from environment variables."""

    # AgentRouter LLM Settings
    agentrouter_api_key: str
    agentrouter_base_url: str
    agentrouter_model: str

    # WooCommerce REST API Credentials
    wc_site_url: str
    wc_consumer_key: str
    wc_consumer_secret: str
    wc_verify_ssl: bool

    # Rate Limiting & Delays (Seconds)
    delay_between_reviews_min: float
    delay_between_reviews_max: float
    delay_between_products_min: float
    delay_between_products_max: float

    # Retries
    max_retries: int
    backoff_factor: float

    # File Paths
    default_csv_path: Path
    state_file_path: Path
    failed_log_path: Path

    # HTTP Client Headers for AgentRouter WAF Verification
    agentrouter_headers: dict[str, str]

    @classmethod
    def load(cls) -> AppConfig:
        """Loads configuration from environment variables and defaults.

        Returns:
            AppConfig: Initialized configuration object.
        """
        api_key = os.getenv("AGENTROUTER_API_KEY", "").strip("\"' ")
        base_url = os.getenv("AGENTROUTER_BASE_URL", "https://agentrouter.org/v1").strip("\"' ")
        model = os.getenv("AGENTROUTER_MODEL", "deepseek-v4-flash").strip("\"' ")

        site_url = os.getenv("WC_SITE_URL", "https://kiachahiye.com").rstrip("/")
        ck = os.getenv("WC_CONSUMER_KEY", "").strip("\"' ")
        cs = os.getenv("WC_CONSUMER_SECRET", "").strip("\"' ")
        verify_ssl = os.getenv("WC_VERIFY_SSL", "true").lower() in ("true", "1", "yes")

        rev_min = float(os.getenv("DELAY_BETWEEN_REVIEWS_MIN", "1.5"))
        rev_max = float(os.getenv("DELAY_BETWEEN_REVIEWS_MAX", "2.5"))
        prod_min = float(os.getenv("DELAY_BETWEEN_PRODUCTS_MIN", "3.5"))
        prod_max = float(os.getenv("DELAY_BETWEEN_PRODUCTS_MAX", "5.0"))

        retries = int(os.getenv("MAX_RETRIES", "3"))
        backoff = float(os.getenv("BACKOFF_FACTOR", "2.0"))

        headers = {
            "User-Agent": "RooCode/3.0.0",
            "x-app": "roo-code",
            "Content-Type": "application/json",
        }

        return cls(
            agentrouter_api_key=api_key,
            agentrouter_base_url=base_url,
            agentrouter_model=model,
            wc_site_url=site_url,
            wc_consumer_key=ck,
            wc_consumer_secret=cs,
            wc_verify_ssl=verify_ssl,
            delay_between_reviews_min=rev_min,
            delay_between_reviews_max=rev_max,
            delay_between_products_min=prod_min,
            delay_between_products_max=prod_max,
            max_retries=retries,
            backoff_factor=backoff,
            default_csv_path=BASE_DIR / "website total products - website total products.csv.csv",
            state_file_path=BASE_DIR / "progress_state.json",
            failed_log_path=BASE_DIR / "failed_reviews.log",
            agentrouter_headers=headers,
        )

    def validate(self, require_wc: bool = True) -> None:
        """Validates critical settings.

        Args:
            require_wc: Whether WooCommerce credentials are mandatory.

        Raises:
            ValueError: If mandatory credentials or parameters are missing.
        """
        errors: list[str] = []
        if not self.agentrouter_api_key:
            errors.append("AGENTROUTER_API_KEY is missing in .env")
        if require_wc:
            if not self.wc_consumer_key:
                errors.append("WC_CONSUMER_KEY is missing in .env")
            if not self.wc_consumer_secret:
                errors.append("WC_CONSUMER_SECRET is missing in .env")

        if errors:
            raise ValueError("Configuration Errors:\n" + "\n".join(f"- {e}" for e in errors))


# Singleton default configuration instance
CONFIG: AppConfig = AppConfig.load()
