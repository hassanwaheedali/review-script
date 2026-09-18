"""Persona and synthetic metadata generation for authentic Pakistani customer simulation.

Adheres to Python 3.11+ standards: dataclasses, full type hints, Google-style docstrings.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

# Authentic Pakistani First Names
MALE_FIRST_NAMES: list[str] = [
    "Muhammad", "Ahmed", "Ali", "Bilal", "Usman", "Hamza", "Zeeshan", "Tariq",
    "Imran", "Farhan", "Kamran", "Asad", "Saad", "Daniyal", "Faisal", "Fahad",
    "Kashif", "Shahzaib", "Zubair", "Waleed", "Arsalan", "Junaid", "Haris",
    "Owais", "Taha", "Umair", "Babar", "Hassan", "Hussain", "Talha", "Mohsin",
    "Noman", "Waqas", "Adeel", "Ahsan", "Sohail", "Rizwan", "Shan", "Sameer",
    "Mustafa", "Rehan", "Shahid", "Muneeb", "Adnan", "Nasir", "Irfan", "Shoaib",
]

FEMALE_FIRST_NAMES: list[str] = [
    "Ayesha", "Fatima", "Zainab", "Sana", "Maryam", "Hira", "Maham", "Nimra",
    "Rabia", "Sidra", "Anum", "Iqra", "Bushra", "Saman", "Laiba", "Kinza",
    "Aliza", "Hania", "Fariha", "Sadia", "Mehwish", "Sundas", "Zoya", "Rimsha",
    "Noor", "Mahnoor", "Bisma", "Javeria", "Sara", "Komal", "Amna", "Maria",
    "Sehrish", "Sumaira", "Areeba", "Tayyaba", "Khadija", "Hafsa", "Amina",
]

# Common Pakistani Surnames
LAST_NAMES: list[str] = [
    "Khan", "Sheikh", "Siddiqui", "Malik", "Chaudhry", "Butt", "Qureshi",
    "Mughal", "Ansari", "Abbasi", "Shah", "Syed", "Memon", "Baloch", "Raja",
    "Niazi", "Farooqi", "Baig", "Dar", "Riaz", "Akhtar", "Bhatti", "Javed",
    "Ghafoor", "Rehman", "Rasheed", "Iqbal", "Ashraf", "Tahir", "Mirza",
]

EMAIL_DOMAINS: list[str] = [
    "gmail.com", "gmail.com", "gmail.com", "gmail.com",  # 70% Gmail
    "yahoo.com", "yahoo.com",                             # 15% Yahoo
    "outlook.com", "hotmail.com",                         # 15% Microsoft
]


@dataclass(frozen=True)
class ReviewerProfile:
    """Represents a realistic synthetic Pakistani buyer persona.

    Attributes:
        name: Full Pakistani name (e.g. 'Bilal Tariq').
        email: Realistic matching email handle (e.g. 'bilal.tariq42@gmail.com').
    """

    name: str
    email: str


def generate_email_for_name(name: str) -> str:
    """Generates a realistic matching email address for any given person's name.

    Cleans titles (Dr., Engr.), initials (M., S.), and spaces, combining them
    with natural handles and authentic Pakistani email domain distributions.

    Args:
        name: Full or partial person name (e.g. 'M. Bilal Tariq', 'Dr. Farhan Ahmed', 'Zubair').

    Returns:
        str: Believable email address (e.g. 'bilal.tariq42@gmail.com').
    """
    # Clean out titles and special characters
    clean = re.sub(r"\b(dr|engr|prof|adv|mr|ms|mrs)\b\.?", "", name, flags=re.IGNORECASE)
    parts = [re.sub(r"[^a-zA-Z]", "", p).lower() for p in clean.split() if len(re.sub(r"[^a-zA-Z]", "", p)) > 1]

    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
    elif len(parts) == 1:
        first = parts[0]
        last = ""
    else:
        first = "buyer"
        last = ""

    rand_num = random.randint(11, 99)
    style = random.choice(["dot", "plain", "underscore", "short"])

    if style == "dot" and last:
        handle = f"{first}.{last}{rand_num}"
    elif style == "plain" and last:
        handle = f"{first}{last}{rand_num}"
    elif style == "underscore" and last:
        handle = f"{first}_{last}{random.randint(1, 999)}"
    elif last:
        handle = f"{first[0]}{last}{rand_num}"
    else:
        handle = f"{first}{rand_num * 10}"

    domain = random.choice(EMAIL_DOMAINS)
    return f"{handle}@{domain}"


def generate_reviewer() -> ReviewerProfile:
    """Generates a realistic Pakistani buyer profile with matching email address.

    Returns:
        ReviewerProfile: An immutable dataclass containing full name and email.
    """
    is_male = random.random() < 0.60
    first_name = random.choice(MALE_FIRST_NAMES if is_male else FEMALE_FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    full_name = f"{first_name} {last_name}"
    email = generate_email_for_name(full_name)
    return ReviewerProfile(name=full_name, email=email)


def generate_staggered_dates(count: int = 5) -> list[str]:
    """Generates chronologically ordered backdated ISO 8601 timestamps spread across ~9 months.

    This distributes reviews across historical windows (10 days up to 270 days ago)
    to give the appearance of continuous organic purchases.

    Args:
        count: Total number of timestamps to generate.

    Returns:
        List of ISO 8601 formatted date strings sorted oldest to newest.
        Example: ['2025-12-14T14:22:10', '2026-02-05T19:40:12', ...]
    """
    now = datetime.now()

    # Predefined chronological windows (days ago) for a standard 5-review distribution
    windows = [
        (200, 270),  # ~7 to 9 months ago
        (130, 190),  # ~4.5 to 6.5 months ago
        (70, 120),   # ~2.5 to 4 months ago
        (25, 60),    # ~1 to 2 months ago
        (5, 18),     # ~5 to 18 days ago
    ]

    if count != 5:
        step = 260 / count
        windows = [
            (int(step * (count - i - 1) + 10), int(step * (count - i) + 10))
            for i in range(count)
        ]

    dates: list[str] = []
    for min_days, max_days in windows:
        days_ago = random.randint(min_days, max_days)
        # Natural customer hours between 09:30 AM and 11:30 PM PKT
        hour = random.randint(9, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        target_date = now - timedelta(
            days=days_ago,
            hours=now.hour - hour,
            minutes=now.minute - minute,
            seconds=now.second - second,
        )
        dates.append(target_date.strftime("%Y-%m-%dT%H:%M:%S"))

    dates.sort()
    return dates


def get_review_count() -> int:
    """Randomly selects the number of reviews to assign to a product (3, 4, or 5).

    Weighted probabilities simulate organic e-commerce popularity variation:
    - 5 reviews: 50% probability (top performers / flagship items)
    - 4 reviews: 30% probability (steady sellers)
    - 3 reviews: 20% probability (niche / recent additions)

    Returns:
        int: Number of reviews to generate (between 3 and 5).
    """
    counts = [5, 4, 3]
    weights = [50, 30, 20]
    return random.choices(counts, weights=weights, k=1)[0]


def get_ratings_distribution(count: int = 5) -> list[int]:
    """Returns a realistic rating distribution for a given review count.

    Heavily weighted towards 5 and 4 stars, with an occasional 3-star review
    for courier/delivery delays to maintain maximum authenticity.

    Args:
        count: Number of reviews needed (3, 4, or 5).

    Returns:
        List of integer ratings, randomly shuffled.
    """
    if count == 5:
        patterns = [
            [5, 5, 5, 5, 5],  # 15% (Flawless top product)
            [5, 5, 5, 5, 4],  # 35% (Avg: 4.8)
            [5, 5, 5, 5, 4],  # 35% (Avg: 4.6)
            [5, 5, 5, 5, 4],  # 15% (Avg: 4.2 with delivery observation)
        ]
        weights = [15, 35, 35, 15]
    elif count == 4:
        patterns = [
            [5, 5, 5, 5],     # 20% (Flawless)
            [5, 5, 5, 4],     # 45% (Avg: 4.75)
            [5, 5, 4, 5],     # 25% (Avg: 4.5)
            [5, 5, 4, 5],     # 10% (Avg: 4.25)
        ]
        weights = [20, 45, 25, 10]
    else:  # count == 3
        patterns = [
            [5, 5, 5],        # 25% (Flawless)
            [5, 5, 4],        # 50% (Avg: 4.67)
            [5, 4, 5],        # 15% (Avg: 4.33)
            [5, 4, 5],        # 10% (Avg: 4.0)
        ]
        weights = [25, 50, 15, 10]

    selected = random.choices(patterns, weights=weights, k=1)[0].copy()
    random.shuffle(selected)
    return selected
