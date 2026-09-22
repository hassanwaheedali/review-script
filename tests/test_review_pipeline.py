"""Comprehensive test suite for the WooCommerce Review Automation Pipeline.

Adheres to python-pro standards: pytest fixtures, mocking, and edge-case testing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import AppConfig
from llm_client import GeneratedReview, LLMClient
from main import (
    JobExecutionStats,
    ProductRecord,
    StateManager,
    load_products_from_csv,
    process_product_pipeline,
)
from persona_data import (
    EMAIL_DOMAINS,
    ReviewerProfile,
    generate_email_for_name,
    generate_reviewer,
    generate_staggered_dates,
    get_ratings_distribution,
    get_review_count,
)
from prompt_templates import (
    SYSTEM_PROMPT,
    build_review_prompt,
    detect_category,
    get_category_vocab,
    is_installation_candidate,
)
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
    """Tests system prompt conforms to DeepSeek-optimized structure with persona-first design."""
    # Core persona-first sections
    assert "<persona>" in SYSTEM_PROMPT
    assert "<voice_calibration>" in SYSTEM_PROMPT
    assert "<product_rules>" in SYSTEM_PROMPT
    assert "<guardrails>" in SYSTEM_PROMPT
    assert "<review_tiers>" in SYSTEM_PROMPT
    assert "<rating_sentiment>" in SYSTEM_PROMPT
    assert "<output_format>" in SYSTEM_PROMPT
    # New sections added for authenticity
    assert "<mobile_typing_reality>" in SYSTEM_PROMPT
    assert "<anti_bot_checklist>" in SYSTEM_PROMPT
    # Key guardrails content preserved
    assert "craftsmanship" in SYSTEM_PROMPT
    assert "installation" in SYSTEM_PROMPT.lower()
    assert "bewi" in SYSTEM_PROMPT.lower()
    # DeepSeek-specific: WhatsApp-style, character archetypes
    assert "whatsapp" in SYSTEM_PROMPT.lower()
    assert "tiktok" in SYSTEM_PROMPT.lower()


def test_detect_category() -> None:
    """Tests product category detection from title keywords."""
    assert detect_category("Haier HR-66B 2.5 Cu Ft Refrigerator") == "refrigerator"
    assert detect_category("WestPoint WF-9216 Hand Blender Set") == "blender"
    assert detect_category("Anex AG-1062 Dry Iron") == "iron"
    assert detect_category("Dawlance 1.5 Ton Inverter AC") == "ac"
    assert detect_category("Some Unknown Product XYZ") == "generic"
    assert detect_category("Philips Hair Straightener") == "straightener"
    assert detect_category("National Electric Kettle") == "kettle"
    assert detect_category("Haier 8 KG Automatic Washing Machine") == "washing_machine"
    # New categories
    assert detect_category("WestPoint WF-2023 1.8 Liters Coffee Maker") == "coffee_maker"
    assert detect_category("ELite Metal Table Fan Copper ETF-30M") == "fan"
    assert detect_category("ELite ETF-003 Evaporative Cooler Tower Fan") == "cooler"
    assert detect_category("WestPoint WF-142 2000 Watts Ceramic Cooker") == "cooker"
    assert detect_category("WestPoint WF-3669 Deluxe High Suction Vacuum Cleaner") == "vacuum_cleaner"
    assert detect_category("Haier HWS 60-50 Spin Dryer") == "washing_machine"
    assert detect_category("TCL 32S51K 32 QLED Smart TV") == "led_tv"
    assert detect_category("ELite EAP-911 Digital 3-in-1 Air Purifier") == "air_purifier"
    # Disambiguation and collision fixes
    assert detect_category("WestPoint WF-2405 Deluxe 750W Stainless Steel Spinner Juicer") == "blender"
    assert detect_category("Dawlance DDW 14952 S INV 14 Place Settings Silver Inverter Dishwasher") == "dishwasher"


def test_get_category_vocab() -> None:
    """Tests focused vocabulary bank retrieval per category."""
    fridge_vocab = get_category_vocab("refrigerator")
    assert "compressor" in fridge_vocab
    assert "cooling" in fridge_vocab

    blender_vocab = get_category_vocab("blender")
    assert "blades" in blender_vocab
    assert "sharp" in blender_vocab

    generic_vocab = get_category_vocab("unknown_category")
    assert "product quality" in generic_vocab

    # New categories have proper vocab
    coffee_vocab = get_category_vocab("coffee_maker")
    assert "coffee" in coffee_vocab
    assert "filter" in coffee_vocab

    fan_vocab = get_category_vocab("fan")
    assert "hawa" in fan_vocab

    dishwasher_vocab = get_category_vocab("dishwasher")
    assert "bartan" in dishwasher_vocab
    assert "Real buyers say:" in dishwasher_vocab

    vacuum_vocab = get_category_vocab("vacuum_cleaner")
    assert "suction" in vacuum_vocab


def test_build_review_prompt() -> None:
    """Tests user prompt construction with product details, target ratings, and tier constraints."""
    # Test strict short-only mode (75-80% catalog distribution)
    prompt_short = build_review_prompt("Haier Refrigerator", "HR-66B", [5, 5, 4, 5, 4], allow_detailed=False)
    assert "<task>" in prompt_short
    assert "<product_title>Haier Refrigerator</product_title>" in prompt_short
    assert "<sku>HR-66B</sku>" in prompt_short
    assert "<target_ratings>[5, 5, 4, 5, 4]</target_ratings>" in prompt_short
    assert "SEEDHI BAAT" in prompt_short
    assert "NO TIER 3" in prompt_short
    assert "<free_installation>" not in prompt_short
    # Category-specific context should be injected
    assert "<product_context>" in prompt_short
    assert "refrigerator" in prompt_short.lower()

    # Test detailed-allowed mode (20-25% catalog distribution)
    prompt_detailed = build_review_prompt("Haier Refrigerator", "HR-66B", [5, 4, 5], allow_detailed=True)
    assert "Tier 3" in prompt_detailed

    # Test free installation directive inclusion
    prompt_with_inst = build_review_prompt(
        "HAIER 8.5KG AUTOMATIC WASHING MACHINE", "HWM85", [5, 5, 4], include_installation=True
    )
    assert "<free_installation>" in prompt_with_inst
    assert "free installation" in prompt_with_inst.lower()

    # Test WestPoint brand logistics constraint
    prompt_westpoint = build_review_prompt(
        "WestPoint WF-9216 Hand Blender Set", "WF-9216", [5, 4, 3], allow_detailed=False
    )
    assert "BANNED" in prompt_westpoint

    # Test customer service directive inclusion
    prompt_with_cs = build_review_prompt(
        "Haier Refrigerator", "HR-66B", [5, 5, 4], include_customer_service=True
    )
    assert "<customer_service>" in prompt_with_cs
    assert "customer service" in prompt_with_cs.lower()

    # Test think_first chain-of-thought anchor (DeepSeek-specific)
    assert "<think_first>" in prompt_short


def test_is_installation_candidate() -> None:
    """Tests accurate classification of major appliances vs small gadgets."""
    # Appliances eligible for free installation (active keywords: ACs and Geysers)
    assert is_installation_candidate("Gree 1.5 Ton Fairy Inverter AC Heat & Cool") is True
    assert is_installation_candidate("Canon 20L Instant Gas Geyser") is True
    assert is_installation_candidate("Dawlance 2 Ton Split AC") is True

    # Other items not eligible for installation under current configuration
    assert is_installation_candidate("WestPoint WF-9216 700ml Deluxe Hand Blender Set") is False
    assert is_installation_candidate("Anex AG-1062 Deluxe Dry Iron") is False
    assert is_installation_candidate("Philips Electric Kettle 1.7L") is False
    assert is_installation_candidate("Braun Series 3 Electric Shaver Trimmer") is False



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


def test_sanitize_review(mock_config: AppConfig) -> None:
    """Tests post-processing sanitizer catches bot artifacts."""
    client = LLMClient(mock_config)

    # Self-correction pattern
    assert "wife" not in client._sanitize_review("Zabardast cheez hai, wife... nai ghar ke liye lia tha")
    assert "ghar walon" in client._sanitize_review("wife ke liye mangwaya")

    # AI buzzwords removed
    cleaned = client._sanitize_review("This product is seamless and delighted me with its craftsmanship")
    assert "seamless" not in cleaned
    assert "delighted" not in cleaned
    assert "craftsmanship" not in cleaned

    # Hindi words removed
    cleaned_hindi = client._sanitize_review("Boht turant aur suvidha wala hai")
    assert "turant" not in cleaned_hindi
    assert "suvidha" not in cleaned_hindi

    # Clean text passes through unchanged
    clean = "Boht achi cheez hai, masala barik pees deta hai"
    assert client._sanitize_review(clean) == clean

    # Double spaces cleaned
    assert "  " not in client._sanitize_review("Achi  cheez   hai")


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


@patch("wc_client.requests.Session.delete")
@patch("wc_client.requests.Session.get")
def test_wc_get_and_delete_reviews(
    mock_get: MagicMock,
    mock_delete: MagicMock,
    mock_config: AppConfig,
) -> None:
    """Tests fetching and deleting product reviews from WooCommerce."""
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = [{"id": 1137, "reviewer": "Sana Javed", "rating": 5}]
    mock_get.return_value = mock_get_resp

    mock_del_resp = MagicMock()
    mock_del_resp.status_code = 200
    mock_delete.return_value = mock_del_resp

    client = WooCommerceReviewClient(mock_config)
    reviews = client.get_product_reviews(10701)
    assert len(reviews) == 1
    assert reviews[0]["id"] == 1137

    deleted = client.delete_review(1137, force=True)
    assert deleted is True



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


# --- Pipeline & Safeguard Approval Gate Tests ---

@patch("main.Prompt.ask")
def test_process_product_pipeline_safeguard_regenerate_then_approve(
    mock_prompt_ask: MagicMock,
    tmp_path: Path,
) -> None:
    """Tests the safeguard approval gate: operator rejects first batch ('n'), then approves second ('y')."""
    mock_prompt_ask.side_effect = ["n", "y"]

    mock_llm = MagicMock(spec=LLMClient)
    batch_1 = [
        GeneratedReview(rating=5, review="Pehli batch review text", name="Ali Khan"),
    ]
    batch_2 = [
        GeneratedReview(rating=4, review="Dusri batch fresh review", name="Usman Tariq"),
    ]
    mock_llm.generate_reviews.side_effect = [batch_1, batch_2]

    state = StateManager(tmp_path / "state.json")
    stats = JobExecutionStats()
    prod = ProductRecord(id=9999, name="Test Dawlance Fridge", sku="DW-99", product_type="simple")

    success = process_product_pipeline(
        prod=prod,
        llm_client=mock_llm,
        wc_client=None,
        is_dry_run=True,
        state=state,
        stats=stats,
        enable_anim=False,
        require_approval=True,
    )

    assert success is True
    # LLM must be called twice because first batch was discarded on operator 'n'
    assert mock_llm.generate_reviews.call_count == 2
    assert stats.processed_products == 1
    assert stats.total_reviews_posted == 0  # In dry run, stats.total_reviews_posted tracks live writes


@patch("main.Prompt.ask")
def test_process_product_pipeline_safeguard_skip(
    mock_prompt_ask: MagicMock,
    tmp_path: Path,
) -> None:
    """Tests the safeguard approval gate when operator chooses to skip product ('s')."""
    mock_prompt_ask.return_value = "s"

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_reviews.return_value = [
        GeneratedReview(rating=5, review="Review to skip", name="Ali Khan"),
    ]

    state = StateManager(tmp_path / "state.json")
    stats = JobExecutionStats()
    prod = ProductRecord(id=8888, name="Skipped Fridge", sku="DW-88", product_type="simple")

    success = process_product_pipeline(
        prod=prod,
        llm_client=mock_llm,
        wc_client=None,
        is_dry_run=True,
        state=state,
        stats=stats,
        enable_anim=False,
        require_approval=True,
    )

    assert success is True
    assert mock_llm.generate_reviews.call_count == 1
    assert stats.processed_products == 0
    assert 8888 not in state.completed_ids


def test_process_product_pipeline_auto_approve(tmp_path: Path) -> None:
    """Tests the pipeline when require_approval=False (auto-approve flag)."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_reviews.return_value = [
        GeneratedReview(rating=5, review="Auto approved review", name="Hamza"),
    ]

    state = StateManager(tmp_path / "state.json")
    stats = JobExecutionStats()
    prod = ProductRecord(id=7777, name="Auto Approved TV", sku="TV-77", product_type="simple")

    success = process_product_pipeline(
        prod=prod,
        llm_client=mock_llm,
        wc_client=None,
        is_dry_run=True,
        state=state,
        stats=stats,
        enable_anim=False,
        require_approval=False,
    )

    assert success is True
    assert mock_llm.generate_reviews.call_count == 1
    assert stats.processed_products == 1

