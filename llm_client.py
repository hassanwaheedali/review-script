"""Robust LLM client for generating authentic customer reviews via AgentRouter.

Adheres to python-pro standards: dataclasses, type annotations (Python 3.11+),
Google-style docstrings, and exponential backoff retry handling.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from config import CONFIG, AppConfig
from persona_data import generate_reviewer
from prompt_templates import SYSTEM_PROMPT, build_review_prompt, is_installation_candidate


@dataclass(frozen=True)
class GeneratedReview:
    """Represents a single AI-generated customer review.

    Attributes:
        name: Reviewer's authentic Pakistani name.
        rating: Integer star rating between 1 and 5.
        review: Conversational review text.
    """

    name: str
    rating: int
    review: str


class LLMClient:
    """Client for generating authentic reviews using AgentRouter."""

    def __init__(self, config: AppConfig = CONFIG) -> None:
        """Initializes the LLM client with configuration settings.

        Args:
            config: Application configuration instance.

        Raises:
            ValueError: If AGENTROUTER_API_KEY is not provided.
        """
        self.config = config
        if not self.config.agentrouter_api_key:
            raise ValueError("AGENTROUTER_API_KEY is not set in config / .env")

        self.endpoint: str = f"{self.config.agentrouter_base_url}/chat/completions"
        self.headers: dict[str, str] = {
            **self.config.agentrouter_headers,
            "Authorization": f"Bearer {self.config.agentrouter_api_key}",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _extract_json(self, raw_text: str) -> list[dict[str, Any]] | None:
        """Cleans and extracts JSON array from raw model text.

        Handles markdown blocks (```json ... ```) or conversational wrappers.

        Args:
            raw_text: Raw response string from the model.

        Returns:
            Extracted list of review dictionaries or None if parsing fails.
        """
        text = raw_text.strip()

        # Remove markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Try direct JSON parsing
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "reviews" in data and isinstance(data["reviews"], list):
                return data["reviews"]
        except json.JSONDecodeError:
            pass

        # Fallback: search for JSON array pattern [...]
        match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        return None

    def generate_reviews(
        self,
        product_name: str,
        sku: str,
        ratings: list[int],
        allow_detailed: bool | None = None,
        include_installation: bool | None = None,
        include_customer_service: bool | None = None,
    ) -> list[GeneratedReview]:
        """Generates tailored customer reviews for a given product matching target ratings.

        Args:
            product_name: Name of the product.
            sku: SKU or model number.
            ratings: Target rating list (e.g. [5, 5, 5, 4, 4]).
            allow_detailed: Whether to allow a detailed review. If None, ~20% chance across catalog.
            include_installation: Whether to highlight free installation in one review.
                If None, ~35% chance for eligible appliance categories, and 0% for non-appliances.
            include_customer_service: Whether to mention customer service in one review.
                If None, ~25% chance across catalog.

        Returns:
            List of GeneratedReview instances.

        Raises:
            RuntimeError: If review generation fails after maximum retries.
        """
        if allow_detailed is None:
            # 20% chance across bulk catalog to include 1 moderate detailed review
            allow_detailed = random.random() < 0.20

        if include_installation is None:
            # Realistic probability: ~35% chance only for eligible home appliances
            include_installation = is_installation_candidate(product_name) and (random.random() < 0.35)

        if include_customer_service is None:
            # Realistic probability: ~25% chance across catalog to feature customer support
            include_customer_service = random.random() < 0.25

        user_prompt = build_review_prompt(
            product_name,
            sku,
            ratings,
            allow_detailed=allow_detailed,
            include_installation=include_installation,
            include_customer_service=include_customer_service,
        )

        payload: dict[str, Any] = {
            "model": self.config.agentrouter_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.75,
            "max_tokens": 4000,  # 4000: reasoning models (deepseek-v4-flash) require budget for reasoning_content + output JSON
        }

        last_error: str | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.session.post(
                    self.endpoint,
                    json=payload,
                    timeout=60,  # 60s: reasoning models take time to think and generate
                )

                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}: {response.text[:150]}"
                    sleep_time = self.config.backoff_factor ** attempt
                    time.sleep(sleep_time)
                    continue

                response.raise_for_status()
                res_data = response.json()

                choice = res_data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "").strip()

                raw_reviews = self._extract_json(content)
                if raw_reviews and len(raw_reviews) > 0:
                    results: list[GeneratedReview] = []
                    for idx, r in enumerate(raw_reviews[:len(ratings)]):
                        r_name = str(r.get("name", "")).strip() or generate_reviewer().name
                        # Always enforce the pre-planned target rating — never trust LLM's self-generated value
                        r_rating = ratings[idx] if idx < len(ratings) else int(r.get("rating", 5))
                        r_text = str(r.get("review", "")).strip()
                        results.append(GeneratedReview(name=r_name, rating=r_rating, review=r_text))

                    if len(results) == len(ratings):
                        return results

                finish_reason = choice.get("finish_reason", "")
                if finish_reason in ("length", "max_tokens"):
                    last_error = f"Token budget exhausted (finish_reason={finish_reason})"
                elif not content:
                    last_error = f"Empty response received from LLM API (finish_reason={finish_reason})"
                else:
                    last_error = f"Malformed review response: {content[:150]}"

            except requests.RequestException as e:
                last_error = str(e)
                time.sleep(self.config.backoff_factor ** attempt)

        raise RuntimeError(
            f"Failed to generate reviews for '{product_name}' after {self.config.max_retries} attempts. Error: {last_error}"
        )
