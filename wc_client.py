"""Production-grade WooCommerce REST API client for publishing product reviews.

Adheres to python-pro standards: dataclasses, full type hints, Google-style docstrings,
and robust error handling with exponential backoff.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from config import CONFIG, AppConfig


@dataclass(frozen=True)
class ReviewPostResult:
    """Represents the outcome of a WooCommerce review submission.

    Attributes:
        success: Whether the review was accepted and created.
        review_id: The ID of the created review if successful.
        product_id: Target product ID.
        rating: Assigned star rating.
        reviewer: Reviewer's name.
        date_created: ISO 8601 creation timestamp.
        error: Error description if creation failed.
        status_code: HTTP status code received.
    """

    success: bool
    review_id: int | None = None
    product_id: int | None = None
    rating: int | None = None
    reviewer: str | None = None
    date_created: str | None = None
    error: str | None = None
    status_code: int | None = None


class WooCommerceReviewClient:
    """Client to interact with the WooCommerce REST API reviews endpoint."""

    def __init__(self, config: AppConfig = CONFIG) -> None:
        """Initializes the WooCommerce REST client.

        Args:
            config: Application configuration instance.
        """
        self.config = config
        self.site_url: str = self.config.wc_site_url.rstrip("/")
        self.endpoint: str = f"{self.site_url}/wp-json/wc/v3/products/reviews"
        self.auth = HTTPBasicAuth(self.config.wc_consumer_key, self.config.wc_consumer_secret)
        self.verify_ssl: bool = self.config.wc_verify_ssl

        self.session = requests.Session()
        self.session.auth = self.auth

    def test_connection(self) -> dict[str, Any]:
        """Tests API credentials by querying existing reviews.

        Returns:
            Dictionary containing 'success' (bool) and 'message' (str).
        """
        try:
            response = self.session.get(
                self.endpoint,
                params={"per_page": 1},
                verify=self.verify_ssl,
                timeout=15,
            )
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Connection to WooCommerce REST API established successfully!",
                }
            if response.status_code in (401, 403):
                return {
                    "success": False,
                    "message": (
                        f"Authentication failed (HTTP {response.status_code}). "
                        "Please verify WC_CONSUMER_KEY and WC_CONSUMER_SECRET in .env."
                    ),
                }
            return {
                "success": False,
                "message": (
                    f"Unexpected response from WooCommerce (HTTP {response.status_code}): "
                    f"{response.text[:200]}"
                ),
            }
        except requests.RequestException as e:
            return {"success": False, "message": f"Connection error: {e}"}

    def post_review(
        self,
        product_id: int,
        reviewer_name: str,
        reviewer_email: str,
        review_text: str,
        rating: int,
        date_created_iso: str,
        verified: bool = True,
    ) -> ReviewPostResult:
        """Posts a single customer review with 'approved' status and backdated timestamp.

        Args:
            product_id: WooCommerce Product ID.
            reviewer_name: Customer full name.
            reviewer_email: Customer email address.
            review_text: Review content.
            rating: Rating between 1 and 5.
            date_created_iso: ISO 8601 formatted date (YYYY-MM-DDTHH:MM:SS).
            verified: Mark as verified owner.

        Returns:
            ReviewPostResult: Typed result of the posting attempt.
        """
        payload: dict[str, Any] = {
            "product_id": int(product_id),
            "review": review_text.strip(),
            "reviewer": reviewer_name.strip(),
            "reviewer_email": reviewer_email.strip(),
            "rating": int(rating),
            "status": "approved",
            "date_created": date_created_iso,
            "verified": verified,
        }

        last_error: str | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.session.post(
                    self.endpoint,
                    json=payload,
                    verify=self.verify_ssl,
                    timeout=20,
                )

                # HTTP 201 Created
                if response.status_code == 201:
                    data = response.json()
                    return ReviewPostResult(
                        success=True,
                        review_id=data.get("id"),
                        product_id=product_id,
                        rating=rating,
                        reviewer=reviewer_name,
                        date_created=date_created_iso,
                        status_code=201,
                    )

                # Duplicate review detected
                if response.status_code == 409 or "comment_duplicate" in response.text:
                    return ReviewPostResult(
                        success=False,
                        error="Duplicate review detected by WooCommerce",
                        status_code=response.status_code,
                    )

                # Rate limiting or temporary server error -> back off and retry
                if response.status_code in (429, 500, 502, 503, 504):
                    sleep_time = self.config.backoff_factor ** attempt
                    time.sleep(sleep_time)
                    continue

                # Client error (e.g. 400 Bad Request, 404 Not Found)
                error_msg = response.text
                try:
                    err_json = response.json()
                    error_msg = str(err_json.get("message", error_msg))
                except (ValueError, requests.exceptions.JSONDecodeError):
                    pass

                return ReviewPostResult(
                    success=False,
                    error=f"HTTP {response.status_code}: {error_msg}",
                    status_code=response.status_code,
                )

            except requests.RequestException as e:
                last_error = str(e)
                time.sleep(self.config.backoff_factor ** attempt)

        return ReviewPostResult(
            success=False,
            error=f"Request failed after {self.config.max_retries} retries: {last_error}",
        )

    def get_product(self, product_id: int) -> dict[str, Any] | None:
        """Fetches product details (including live permalink) from WooCommerce.

        Args:
            product_id: Target product ID.

        Returns:
            dict containing product metadata or None if request fails.
        """
        try:
            url = f"{self.site_url}/wp-json/wc/v3/products/{product_id}"
            resp = self.session.get(url, verify=self.verify_ssl, timeout=10)
            if resp.status_code == 200:
                data: dict[str, Any] = resp.json()
                return data
        except requests.RequestException:
            pass
        return None

    def get_product_reviews(self, product_id: int) -> list[dict[str, Any]]:
        """Fetches all customer reviews for a given product ID from WooCommerce.

        Args:
            product_id: Target product ID.

        Returns:
            List of review dictionaries returned by WooCommerce REST API.
        """
        try:
            resp = self.session.get(
                self.endpoint,
                params={"product": str(product_id), "per_page": "100"},
                verify=self.verify_ssl,
                timeout=15,
            )
            if resp.status_code == 200:
                data: list[dict[str, Any]] = resp.json()
                return data
        except requests.RequestException:
            pass
        return []

    def delete_review(self, review_id: int, force: bool = True) -> bool:
        """Deletes a review by its ID from WooCommerce.

        Args:
            review_id: ID of the review to delete.
            force: Whether to permanently delete (True) or move to trash (False).

        Returns:
            True if deletion succeeded (HTTP 200), False otherwise.
        """
        try:
            url = f"{self.endpoint}/{review_id}"
            resp = self.session.delete(
                url,
                params={"force": "true" if force else "false"},
                verify=self.verify_ssl,
                timeout=15,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

