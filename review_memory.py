"""Production-grade Review Memory and Anti-Repetition Engine.

Adheres to python-pro standards: Python 3.11+ type annotations, dataclasses,
collections.deque ring buffers, atomic JSON persistence, and Roman Urdu normalization.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Common Roman Urdu phonetic and spelling normalization dictionary
ROMAN_URDU_NORMALIZATION_MAP: dict[str, str] = {
    "hy": "hai",
    "hay": "hai",
    "h": "hai",
    "haa": "hai",
    "nai": "nahi",
    "nahe": "nahi",
    "ni": "nahi",
    "bht": "boht",
    "bohot": "boht",
    "bahout": "boht",
    "thek": "theek",
    "thk": "theek",
    "zyada": "ziada",
    "zda": "ziada",
    "shukriya": "thanks",
    "thnx": "thanks",
    "thanku": "thanks",
    "shukria": "thanks",
    "k": "ke",
    "ki": "ke",
    "ka": "ke",
    "b": "bhi",
    "bi": "bhi",
    "pr": "par",
    "achi": "acha",
    "achaa": "acha",
    "achy": "acha",
    "ache": "acha",
    "recived": "received",
    "recommnded": "recommended",
}


def normalize_roman_urdu(text: str) -> list[str]:
    """Tokenizes and normalizes Roman Urdu text for accurate duplicate detection.

    Args:
        text: Raw review string.

    Returns:
        List of normalized lowercase tokens.
    """
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = cleaned.split()
    return [ROMAN_URDU_NORMALIZATION_MAP.get(tok, tok) for tok in tokens if tok]


def compute_jaccard_similarity(tokens1: list[str], tokens2: list[str]) -> float:
    """Computes Jaccard word-set similarity between two token sequences.

    Args:
        tokens1: First token list.
        tokens2: Second token list.

    Returns:
        Float value between 0.0 and 1.0.
    """
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def has_shared_ngram(tokens1: list[str], tokens2: list[str], n: int = 4) -> bool:
    """Checks if two token lists share any consecutive n-gram sequence.

    Useful for catching verbatim phrase copying like 'zero cut bilkul clean'.

    Args:
        tokens1: First token list.
        tokens2: Second token list.
        n: Size of n-gram to check (default 4).

    Returns:
        True if at least one identical n-gram sequence exists in both.
    """
    if len(tokens1) < n or len(tokens2) < n:
        return False
    ngrams1 = {tuple(tokens1[i : i + n]) for i in range(len(tokens1) - n + 1)}
    for i in range(len(tokens2) - n + 1):
        if tuple(tokens2[i : i + n]) in ngrams1:
            return True
    return False


@dataclass(frozen=True)
class ReviewMemoryRecord:
    """Structured record of an approved and posted customer review.

    Attributes:
        product_id: WooCommerce Product ID.
        product_name: Title of the product.
        category: Detected category key.
        rating: Assigned star rating.
        review_text: Conversational review text.
        timestamp: ISO format timestamp of generation.
    """

    product_id: int
    product_name: str
    category: str
    rating: int
    review_text: str
    timestamp: str


class ReviewMemory:
    """Category-partitioned sliding window memory engine for LLM review generation."""

    def __init__(
        self,
        storage_path: Path,
        max_category_history: int = 15,
        max_global_history: int = 60,
    ) -> None:
        """Initializes the ReviewMemory instance.

        Args:
            storage_path: Path to the JSON persistence file.
            max_category_history: Max reviews retained per product category.
            max_global_history: Max historical records retained in total.
        """
        self.storage_path = storage_path
        self.max_category_history = max_category_history
        self.max_global_history = max_global_history

        # Category-partitioned sliding ring buffers: category -> deque[review_text]
        self._category_buffers: dict[str, deque[str]] = {}
        # Global record list for full persistence
        self._global_records: list[ReviewMemoryRecord] = []
        # In-memory session blacklist for rejected reviews (when operator presses 'n')
        self._session_rejected: set[str] = set()

        self.load()

    def _get_category_buffer(self, category: str) -> deque[str]:
        """Gets or initializes the ring buffer for a given category."""
        if category not in self._category_buffers:
            self._category_buffers[category] = deque(maxlen=self.max_category_history)
        return self._category_buffers[category]

    def load(self) -> None:
        """Loads historical reviews from disk if the storage file exists."""
        if not self.storage_path.exists():
            return

        try:
            with self.storage_path.open("r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)

            records_data = data.get("records", [])
            for item in records_data:
                record = ReviewMemoryRecord(
                    product_id=int(item.get("product_id", 0)),
                    product_name=str(item.get("product_name", "")),
                    category=str(item.get("category", "generic")),
                    rating=int(item.get("rating", 5)),
                    review_text=str(item.get("review_text", "")),
                    timestamp=str(item.get("timestamp", "")),
                )
                self._global_records.append(record)
                buf = self._get_category_buffer(record.category)
                buf.append(record.review_text)

            # Keep global records within limits
            if len(self._global_records) > self.max_global_history:
                self._global_records = self._global_records[-self.max_global_history :]

        except (OSError, json.JSONDecodeError):
            # Gracefully handle corrupted files by starting with clean memory
            self._global_records.clear()
            self._category_buffers.clear()

    def save(self) -> None:
        """Persists the memory state atomically to disk."""
        data = {
            "version": "1.0.0",
            "last_saved": datetime.now().astimezone().isoformat(),
            "records": [asdict(rec) for rec in self._global_records[-self.max_global_history :]],
        }

        # Atomic write pattern using tempfile in same directory
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.storage_path.with_suffix(".tmp")
        try:
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.storage_path)
        except OSError:
            # Fallback direct write if atomic replace encounters platform locks
            with self.storage_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def add_approved_reviews(
        self,
        product_id: int,
        product_name: str,
        category: str,
        reviews: list[Any],
    ) -> None:
        """Adds approved reviews to both category ring buffer and global persistence.

        Args:
            product_id: WooCommerce product ID.
            product_name: Full product name.
            category: Product category key.
            reviews: List of GeneratedReview instances.
        """
        now_iso = datetime.now().astimezone().isoformat()
        buf = self._get_category_buffer(category)

        for rev in reviews:
            rev_text = getattr(rev, "review", str(rev)).strip()
            rev_rating = getattr(rev, "rating", 5)

            if not rev_text:
                continue

            buf.append(rev_text)
            record = ReviewMemoryRecord(
                product_id=product_id,
                product_name=product_name,
                category=category,
                rating=rev_rating,
                review_text=rev_text,
                timestamp=now_iso,
            )
            self._global_records.append(record)

        if len(self._global_records) > self.max_global_history:
            self._global_records = self._global_records[-self.max_global_history :]

        self.save()

    def add_rejected_reviews(self, reviews: list[Any]) -> None:
        """Adds discarded reviews to the in-memory session blacklist.

        Ensures that when an operator rejects a batch via 'n', the LLM will
        not regenerate the same rejected text on the retry attempt.

        Args:
            reviews: List of GeneratedReview instances discarded by the user.
        """
        for rev in reviews:
            rev_text = getattr(rev, "review", str(rev)).strip()
            if rev_text:
                self._session_rejected.add(rev_text)

    def get_anti_repetition_context(self, category: str, limit: int = 8) -> list[str]:
        """Retrieves prioritized recent reviews to inject into the LLM prompt.

        Prioritizes category-specific recent reviews to preserve token efficiency.
        If category history has fewer than limit entries, supplements with general
        recent reviews without exceeding the limit.

        Args:
            category: Detected product category.
            limit: Maximum review strings to return (default 8).

        Returns:
            List of distinct recent review texts.
        """
        results: list[str] = []
        seen: set[str] = set()

        # 1. Prioritize category-specific memory
        if category in self._category_buffers:
            for text in reversed(self._category_buffers[category]):
                if text not in seen:
                    results.append(text)
                    seen.add(text)
                if len(results) >= limit:
                    return results

        # 2. Add rejected reviews from current session (critical for 'n' retries)
        for text in self._session_rejected:
            if text not in seen:
                results.append(text)
                seen.add(text)
            if len(results) >= limit:
                return results

        # 3. Supplement with recent global records if still under limit
        for rec in reversed(self._global_records):
            if rec.review_text not in seen:
                results.append(rec.review_text)
                seen.add(rec.review_text)
            if len(results) >= limit:
                break

        return results

    def check_review_similarity(
        self,
        candidate_text: str,
        category: str,
        existing_batch: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Evaluates whether a candidate review is an unacceptable duplicate.

        Implements length-adaptive thresholds:
        - Short reviews (< 10 words): 85% Jaccard threshold (allows authentic brief phrasing).
        - Longer reviews (>= 10 words): 50% Jaccard threshold or 4-gram verbatim copy.

        Checks against:
        1. Current in-progress batch (ensures diversity within the same product).
        2. Session rejected reviews.
        3. Category memory buffer.

        Args:
            candidate_text: Review string to evaluate.
            category: Product category key.
            existing_batch: Other review texts in the current batch.

        Returns:
            Tuple of (is_duplicate: bool, reason: str).
        """
        candidate_tokens = normalize_roman_urdu(candidate_text)
        if not candidate_tokens:
            return False, ""

        is_short = len(candidate_tokens) < 10
        jaccard_threshold = 0.85 if is_short else 0.50

        # Pool of texts to check against
        comparison_pool: list[tuple[str, str]] = []

        # In-batch checks
        if existing_batch:
            for b_text in existing_batch:
                comparison_pool.append((b_text, "in-batch review"))

        # Session rejected checks
        for r_text in self._session_rejected:
            comparison_pool.append((r_text, "previously discarded review"))

        # Category memory checks
        if category in self._category_buffers:
            for c_text in self._category_buffers[category]:
                comparison_pool.append((c_text, "recent category review"))

        for existing_text, source in comparison_pool:
            existing_tokens = normalize_roman_urdu(existing_text)
            if not existing_tokens:
                continue

            # Exact normalized string match
            if candidate_tokens == existing_tokens:
                return True, f"Exact match with {source}: '{existing_text}'"

            # Jaccard word set similarity
            sim = compute_jaccard_similarity(candidate_tokens, existing_tokens)
            if sim >= jaccard_threshold:
                return True, (
                    f"High token similarity ({sim:.0%}) with {source}: '{existing_text}'"
                )

            # Consecutive 4-gram verbatim copy (only for longer reviews)
            if not is_short and has_shared_ngram(candidate_tokens, existing_tokens, n=4):
                return True, f"Shared 4-gram phrase with {source}: '{existing_text}'"

        return False, ""
