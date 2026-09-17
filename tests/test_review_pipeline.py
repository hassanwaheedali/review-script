"""Comprehensive test suite for the WooCommerce Review Automation Pipeline.

Adheres to python-pro standards: pytest fixtures, mocking, and edge-case testing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import AppConfig
from llm_client import LLMClient
from main import StateManager, load_products_from_csv
from persona_data import (
    EMAIL_DOMAINS,
    ReviewerProfile,
    generate_email_for_name,
    generate_reviewer,
    generate_staggered_dates,
    get_ratings_distribution,
    get_review_count,
)
from prompt_templates import SYSTEM_PROMPT, build_review_prompt
from wc_client import ReviewPostResult, WooCommerceReviewClient

# --- Fixtures ---

@pytest.fixture
def mock_config(tmp_path: Path) -> AppConfig:
    """Fixture providing an AppConfig instance with isolated temporary paths."""
    return AppConfig(
        agentrouter_api_key="sk-test-key",
        agentrouter_base_url="https://agentrouter.org/v1",
        agentrouter_model="deepseek-v4-flash",
        wc_site_url="https://teststore.com",
        wc_consumer_key="ck_test",
        wc_consumer_secret="cs_test",
        wc_verify_ssl=False,
        delay_between_reviews_min=0.01,
        delay_between_reviews_max=0.02,
        delay_between_products_min=0.01,
        delay_between_products_max=0.02,
        max_retries=2,
        backoff_factor=1.1,
        default_csv_path=tmp_path / "test_products.csv",
        state_file_path=tmp_path / "test_state.json",
        failed_log_path=tmp_path / "test_failed.log",
        agentrouter_headers={"User-Agent": "RooCode/3.0.0", "x-app": "roo-code"},
    )


# --- Persona & Metadata Tests ---

def test_generate_reviewer() -> None:
    """Tests synthetic reviewer generation produces valid names and emails."""
    profile = generate_reviewer()
    assert isinstance(profile, ReviewerProfile)
    assert len(profile.name.split()) >= 2
    assert "@" in profile.email
    domain = profile.email.split("@")[1]
    assert domain in EMAIL_DOMAINS


def test_generate_email_for_name() -> None:
    """Tests dynamic email creation for various Pakistani name formats."""
    # Full name
    e1 = generate_email_for_name("M. Bilal Tariq")
    assert "@" in e1
    assert "bilal" in e1 or "tariq" in e1

    # Title with name
    e2 = generate_email_for_name("Dr. Farhan Ahmed")
    assert "@" in e2
    assert "dr" not in e2.split("@")[0]  # clean title

    # Single name
    e3 = generate_email_for_name("Zubair")
    assert "@" in e3
    assert "zubair" in e3


def test_generate_staggered_dates() -> None:
    """Tests staggered date generation produces chronologically ordered ISO timestamps."""
    dates = generate_staggered_dates(count=5)
    assert len(dates) == 5

    parsed_dates: list[datetime] = []
    for d_str in dates:
        dt = datetime.strptime(d_str, "%Y-%m-%dT%H:%M:%S")
        parsed_dates.append(dt)

    # Must be sorted in ascending order (oldest to newest)
    assert parsed_dates == sorted(parsed_dates)

    # Newest date should be recent (within past 30 days)
    days_ago_newest = (datetime.now() - parsed_dates[-1]).days
    assert 0 <= days_ago_newest <= 30

    # Oldest date should be historical (within past 300 days)
    days_ago_oldest = (datetime.now() - parsed_dates[0]).days
    assert 150 <= days_ago_oldest <= 300


def test_get_review_count() -> None:
    """Tests that review count is dynamically distributed between 3, 4, and 5."""
    counts = {get_review_count() for _ in range(50)}
    assert counts.issubset({3, 4, 5})
    assert len(counts) > 1  # Verify randomness occurs


def test_get_ratings_distribution() -> None:
    """Tests ratings distribution for counts 3, 4, and 5 biased toward 5 & 4."""
    for c in [3, 4, 5]:
        ratings = get_ratings_distribution(count=c)
        assert len(ratings) == c
        assert all(3 <= r <= 5 for r in ratings)
        assert sum(ratings) / len(ratings) >= 4.0


# --- Prompt Engineering Tests ---

def test_system_prompt_structure() -> None:
    """Tests system prompt conforms to prompt-engineer XML standard."""
    assert "<system_instructions>" in SYSTEM_PROMPT
    assert "<role>" in SYSTEM_PROMPT
    assert "<persona_and_style>" in SYSTEM_PROMPT
    assert "<negative_constraints>" in SYSTEM_PROMPT
    assert "<output_format>" in SYSTEM_PROMPT
    assert "craftsmanship" in SYSTEM_PROMPT  # explicit negative constraint mentioned


def test_build_review_prompt() -> None:
    """Tests user prompt construction with product details and target ratings."""
    prompt = build_review_prompt("Haier Refrigerator", "HR-66B", [5, 5, 4, 5, 4])
    assert "<task>" in prompt
    assert "<product_title>Haier Refrigerator</product_title>" in prompt
    assert "<sku>HR-66B</sku>" in prompt
    assert "<target_ratings>[5, 5, 4, 5, 4]</target_ratings>" in prompt


# --- LLM Client JSON Extraction Tests ---

def test_llm_json_extraction(mock_config: AppConfig) -> None:
    """Tests JSON extraction across plain JSON, markdown fences, and embedded blocks."""
    client = LLMClient(mock_config)

    # 1. Clean JSON array
    clean_json = '[{"rating": 5, "review": "Zabardast"}, {"rating": 4, "review": "Acha hai"}]'
    res = client._extract_json(clean_json)
    assert res is not None
    assert len(res) == 2
    assert res[0]["rating"] == 5

    # 2. Markdown wrapped JSON
    md_json = '```json\n[{"rating": 5, "review": "Zabardast"}]\n```'
    res = client._extract_json(md_json)
    assert res is not None
    assert len(res) == 1

    # 3. Dict containing reviews key
    dict_json = '{"reviews": [{"rating": 5, "review": "Good"}]}'
    res = client._extract_json(dict_json)
    assert res is not None
    assert len(res) == 1

    # 4. Invalid text
    bad_text = "This is not json at all."
    assert client._extract_json(bad_text) is None


# --- WooCommerce Client Tests ---

@patch("wc_client.requests.Session.post")
def test_wc_post_review_success(mock_post: MagicMock, mock_config: AppConfig) -> None:
    """Tests successful review posting to WooCommerce."""
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"id": 1234, "status": "approved"}
    mock_post.return_value = mock_resp

    client = WooCommerceReviewClient(mock_config)
    result = client.post_review(
        product_id=10701,
        reviewer_name="Bilal Tariq",
        reviewer_email="bilal@gmail.com",
        review_text="Boht achi cheez hai",
        rating=5,
        date_created_iso="2026-03-01T12:00:00",
    )

    assert isinstance(result, ReviewPostResult)
    assert result.success is True
    assert result.review_id == 1234
    assert result.status_code == 201


@patch("wc_client.requests.Session.post")
def test_wc_post_review_duplicate(mock_post: MagicMock, mock_config: AppConfig) -> None:
    """Tests handling of duplicate review responses from WooCommerce."""
    mock_resp = MagicMock()
    mock_resp.status_code = 409
    mock_resp.text = "comment_duplicate"
    mock_post.return_value = mock_resp

    client = WooCommerceReviewClient(mock_config)
    result = client.post_review(
        product_id=10701,
        reviewer_name="Bilal Tariq",
        reviewer_email="bilal@gmail.com",
        review_text="Boht achi cheez hai",
        rating=5,
        date_created_iso="2026-03-01T12:00:00",
    )

    assert result.success is False
    assert "Duplicate" in (result.error or "")


# --- CSV & State Management Tests ---

def test_load_products_from_csv(tmp_path: Path) -> None:
    """Tests CSV parsing extracts valid ProductRecord items."""
    csv_file = tmp_path / "products.csv"
    csv_file.write_text(
        "ID,Type,SKU,Name\n"
        "101,simple,SKU-A,Product One\n"
        "102,variable,SKU-B,Product Two\n"
        ",simple,,Empty ID\n"
        "abc,simple,,Invalid ID\n",
        encoding="utf-8-sig",
    )

    products = load_products_from_csv(csv_file)
    assert len(products) == 2
    assert products[0].id == 101
    assert products[0].name == "Product One"
    assert products[1].id == 102


def test_state_manager(tmp_path: Path) -> None:
    """Tests state persistence and resumption functionality."""
    state_file = tmp_path / "state.json"
    manager = StateManager(state_file)
    assert len(manager.completed_ids) == 0

    # Mark completed
    manager.mark_completed(101, 5)
    manager.mark_completed(102, 5)
    assert 101 in manager.completed_ids
    assert manager.total_reviews_posted == 10

    # Reload fresh instance
    manager2 = StateManager(state_file)
    assert 101 in manager2.completed_ids
    assert 102 in manager2.completed_ids
    assert manager2.total_reviews_posted == 10
