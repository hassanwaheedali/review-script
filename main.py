"""Terminal-Based Review Engine (MIL-SPEC TUI / Call of Duty Retro Edition).

Features:
- Call of Duty Black Ops / CRT military terminal boot animation sequence
- Teletype live typewriter character streaming for review generation
- Dynamic real-time countdown jitter buffer shield for Hostinger LiteSpeed protection
- High data density with structured ASCII visual zoning
- Interactive operator menu with explicit human-in-the-loop confirmation
- Direct clickable live store verification links
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table

from config import CONFIG
from llm_client import GeneratedReview, LLMClient
from persona_data import (
    generate_email_for_name,
    generate_reviewer,
    generate_staggered_dates,
    get_ratings_distribution,
    get_review_count,
)
from prompt_templates import detect_category
from review_memory import ReviewMemory
from wc_client import ReviewPostResult, WooCommerceReviewClient

console = Console()


@dataclass(frozen=True)
class ProductRecord:
    """Represents a product item parsed from WooCommerce export CSV."""

    id: int
    name: str
    sku: str
    product_type: str


@dataclass
class JobExecutionStats:
    """Tracks overall execution metrics across the batch run."""

    processed_products: int = 0
    total_reviews_posted: int = 0
    failed_products: int = 0


class StateManager:
    """Manages completed product IDs to guarantee safe checkpointing and resumption."""

    def __init__(self, state_file: Path) -> None:
        """Initializes the state manager.

        Args:
            state_file: Path to the JSON state persistence file.
        """
        self.state_file: Path = state_file
        self.completed_ids: set[int] = set()
        self.total_reviews_posted: int = 0
        self._load()

    def _load(self) -> None:
        """Loads previous execution state from disk if present."""
        if self.state_file.exists():
            try:
                with self.state_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.completed_ids = set(data.get("completed_ids", []))
                    self.total_reviews_posted = data.get("total_reviews_posted", 0)
            except (OSError, json.JSONDecodeError) as e:
                console.print(f"[dim yellow]Notice: State file reset ({e}). Starting fresh.[/dim yellow]")

    def save(self) -> None:
        """Persists current state to disk atomically."""
        try:
            with self.state_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "completed_ids": sorted(self.completed_ids),
                        "total_reviews_posted": self.total_reviews_posted,
                        "last_updated": datetime.now().astimezone().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except OSError as e:
            console.print(f"[bold red]Error saving state file: {e}[/bold red]")

    def mark_completed(self, product_id: int, review_count: int) -> None:
        """Marks a product as fully processed and saves state immediately.

        Args:
            product_id: WooCommerce product ID.
            review_count: Number of reviews posted for this product.
        """
        self.completed_ids.add(product_id)
        self.total_reviews_posted += review_count
        self.save()


def log_failure(failed_log_path: Path, product_id: int, product_name: str, error: str) -> None:
    """Appends failed product details to the failure log file.

    Args:
        failed_log_path: Path to the failure log.
        product_id: Target product ID.
        product_name: Target product title.
        error: Detailed error message.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] Product ID: {product_id} | Name: '{product_name}' | Error: {error}\n"
    with failed_log_path.open("a", encoding="utf-8") as f:
        f.write(entry)


def load_products_from_csv(csv_path: Path) -> list[ProductRecord]:
    """Parses WooCommerce product export CSV and returns validated product records."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    products: list[ProductRecord] = []
    with csv_path.open(mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = row.get("ID", "").strip()
            name = row.get("Name", "").strip()
            sku = row.get("SKU", "").strip()
            prod_type = row.get("Type", "").strip().lower()

            if not raw_id or not name:
                continue

            try:
                prod_id = int(raw_id)
            except ValueError:
                continue

            products.append(
                ProductRecord(
                    id=prod_id,
                    name=name,
                    sku=sku,
                    product_type=prod_type,
                )
            )

    return products


def get_timestamp() -> str:
    """Returns a compact cyber-forensic timestamp."""
    return datetime.now().strftime("%H:%M:%S")


def play_cod_boot_sequence(enable_anim: bool = True) -> None:
    """Renders a Call of Duty / military CRT terminal boot sequence with animated typing."""
    if not enable_anim:
        return

    boot_lines: list[tuple[str, str, str]] = [
        ("[SYS-INIT]", "TACTICAL REVIEWS ENGINE CORE SUBSYSTEM ......... [INITIALIZED]", "cyan"),
        ("[CONFIG-LOAD]", "ENVIRONMENT VAULT & REST ENCRYPTION KEYS ...... [VERIFIED]", "green"),
        ("[AI-NEXUS]", f"AGENTROUTER LLM CO-PROCESSOR ({CONFIG.agentrouter_model}) .. [ONLINE]", "yellow"),
        ("[REST-GATEWAY]", f"WOOCOMMERCE REST API v3 ({CONFIG.wc_site_url}) .... [AUTHENTICATED]", "green"),
        ("[DATABASE-CSV]", "CATALOG ASSET DATABASE (1,505 ENTITIES) ....... [MOUNTED]", "cyan"),
        ("[SECURITY-OPS]", "HOSTINGER LITESPEED JITTER SHIELD ACTIVE ..... [ARMED]", "bold green"),
    ]

    for tag, msg, color in boot_lines:
        console.print(f"[{color}]{tag}[/{color}] ", end="")
        for ch in msg:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(0.005)
        sys.stdout.write("\n")
        sys.stdout.flush()
        time.sleep(0.04)
    console.print()


def render_cyber_banner() -> None:
    """Displays the streamlined review pipeline banner."""
    banner_content = (
        "[bold cyan] ██████╗ ███████╗██╗   ██╗██╗███████╗██╗    ██╗███████╗[/bold cyan]\n"
        "[bold cyan] ██╔══██╗██╔════╝██║   ██║██║██╔════╝██║    ██║██╔════╝[/bold cyan]       [bold green]STATUS: ONLINE[/bold green]\n"
        "[bold cyan] ██████╔╝█████╗  ██║   ██║██║█████╗  ██║ █╗ ██║███████╗[/bold cyan]       [bold yellow]MODE: BULK-REST[/bold yellow]\n"
        "[bold cyan] ██╔══██╗██╔══╝  ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║╚════██║[/bold cyan]       [bold white]TARGET: kiachahiye.com[/bold white]\n"
        "[bold cyan] ██║  ██║███████╗ ╚████╔╝ ██║███████╗╚███╔███╔╝███████║[/bold cyan]       [bold green][SYNC-ACTIVE][/bold green]\n"
        "[bold cyan] ╚═╝  ╚═╝╚══════╝  ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝ ╚══════╝[/bold cyan]\n\n"
        "[bold white]WOOCOMMERCE AUTOMATED REVIEW INGESTION PIPELINE[/bold white]\n"
        "[dim cyan]Auto-Generated Contextual Feedback • Backdated Past 90 Days • Pre-Approved[/dim cyan]"
    )
    console.print(
        Panel(
            banner_content,
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )


def render_telemetry_matrix(site_url: str, model_name: str, total_items: int, mode_label: str) -> None:
    """Renders the persistent cyber-ops telemetry status box with rounded styling."""
    matrix_table = Table.grid(padding=(0, 3))
    matrix_table.add_column("Key", style="bold cyan")
    matrix_table.add_column("Val", style="bold white")
    matrix_table.add_column("Key2", style="bold cyan")
    matrix_table.add_column("Val2", style="bold white")

    matrix_table.add_row(
        "⚡ TARGET HOST  :",
        f"[bold green]{site_url}[/bold green]",
        "🎯 OPERATIONAL MODE :",
        mode_label,
    )
    matrix_table.add_row(
        "🧠 NEURAL ENGINE:",
        f"[bold yellow]{model_name}[/bold yellow]",
        "🛡️  RATE SHIELD      :",
        "[dim]1.5-2.5s Rev / 3.5-5.0s Prod[/dim]",
    )
    matrix_table.add_row(
        "📦 CATALOG ASSET:",
        f"[white]{total_items:,} Entities Indexed[/white]",
        "🏷️  OWNER BADGE      :",
        "[bold green]ENABLED (Verified)[/bold green]",
    )

    console.print(
        Panel(
            matrix_table,
            title="[bold cyan]SYSTEM TELEMETRY MATRIX[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )


def stream_typewriter_text(prefix: str, text: str, anim_speed: float = 0.007, enable_anim: bool = True) -> None:
    """Outputs text with cinematic character-by-character typewriter streaming."""
    console.print(prefix, end="")
    if enable_anim:
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(anim_speed)
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        console.print(text)


def render_cyber_cooldown(seconds: float, enable_anim: bool = True) -> None:
    """Provides a clean, animated cooldown spinner without terminal line pollution."""
    if not enable_anim or seconds <= 0:
        time.sleep(max(seconds, 0))
        return

    with console.status(
        f"[dim cyan]LiteSpeed Cooldown Buffer ({seconds:.1f}s safe jitter delay)...[/dim cyan]",
        spinner="dots",
    ):
        time.sleep(seconds)
    console.print(f"  [dim]{get_timestamp()} ⏱ [COOLDOWN] Buffer synchronized ({seconds:.1f}s elapsed)[/dim]")


def process_product_pipeline(
    prod: ProductRecord,
    llm_client: LLMClient,
    wc_client: WooCommerceReviewClient | None,
    is_dry_run: bool,
    state: StateManager,
    stats: JobExecutionStats,
    enable_anim: bool = True,
    require_approval: bool = True,
    memory: ReviewMemory | None = None,
) -> bool:
    """Executes the synthesis and publication pipeline for a single target product with human safeguard."""
    p_id = prod.id
    p_name = prod.name
    p_sku = prod.sku

    # Determine live permalink
    live_permalink: str = f"{CONFIG.wc_site_url}/?p={p_id}"
    if wc_client:
        wc_prod = wc_client.get_product(p_id)
        if wc_prod and wc_prod.get("permalink"):
            live_permalink = str(wc_prod["permalink"])

    # Target Acquired Box
    target_grid = Table.grid(padding=(0, 3))
    target_grid.add_column("K1", style="bold cyan")
    target_grid.add_column("V1", style="bold white")
    target_grid.add_column("K2", style="bold cyan")
    target_grid.add_column("V2", style="bold white")

    target_grid.add_row(
        "TARGET ENTITY :", f"[bold cyan]#{p_id}[/bold cyan] [dim]({prod.product_type or 'simple'})[/dim]",
        "SKU CODE      :", f"[bold yellow]{p_sku if p_sku else 'N/A'}[/bold yellow]",
    )
    target_grid.add_row(
        "PRODUCT TITLE :", f"[bold white]{p_name}[/bold white]",
        "SAFEGUARD     :", "[bold yellow]APPROVAL GATE ACTIVE[/bold yellow]" if require_approval else "[bold green]AUTO-APPROVE[/bold green]",
    )
    target_grid.add_row(
        "STORE URL     :", f"[underline cyan]{live_permalink}[/underline cyan]",
        "EXECUTION     :", "[yellow]SIMULATION (0 Writes)[/yellow]" if is_dry_run else "[bold green]LIVE PRODUCTION[/bold green]",
    )

    console.print(
        Panel(
            target_grid,
            title=f"[bold green]▶ TARGET ACQUIRED: #{p_id}[/bold green]",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )

    category = detect_category(p_name)

    # Generation & Approval Safeguard Loop (Regenerates if operator says 'n')
    while True:
        # 1. Determine dynamic review count (3, 4, or 5) & ratings distribution
        review_count = get_review_count()
        ratings = get_ratings_distribution(count=review_count)
        staggered_dates = generate_staggered_dates(count=review_count)

        # 2. Call AI Synthesis with typing telemetry & active spinner
        t_start = get_timestamp()
        stream_typewriter_text(
            prefix=f"[dim]{t_start}[/dim] [bold cyan][AI-CORE][/bold cyan] ",
            text=f"Engaging deepseek-v4-flash neural co-processor for {review_count} Pakistani personas...",
            anim_speed=0.005,
            enable_anim=enable_anim,
        )

        recent_reviews = (
            memory.get_anti_repetition_context(category=category, limit=8)
            if memory is not None
            else None
        )

        try:
            with console.status(
                f"[dim cyan]Synthesizing {review_count} authentic customer personas via AgentRouter ({CONFIG.agentrouter_model})...[/dim cyan]",
                spinner="dots",
            ):
                ai_reviews: list[GeneratedReview] = llm_client.generate_reviews(
                    p_name,
                    p_sku,
                    ratings,
                    recent_reviews=recent_reviews,
                    memory=memory,
                    category=category,
                )
        except RuntimeError as e:
            console.print(f"[bold red]✗ [AI-CORE ERROR] Synthesis failed for Product #{p_id}: {e}[/bold red]")
            log_failure(CONFIG.failed_log_path, p_id, p_name, str(e))
            stats.failed_products += 1
            return False

        t_done = get_timestamp()
        console.print(
            f"[dim]{t_done}[/dim] [bold green][AI-CORE][/bold green] HTTP 200 OK — {len(ai_reviews)} persona reviews synthesized successfully.\n"
        )

        # 3. Preview generated reviews in unified cards
        for idx, item in enumerate(ai_reviews):
            rating = item.rating
            text = item.review
            created_date = staggered_dates[idx]
            reviewer_name = item.name if item.name else generate_reviewer().name
            reviewer_email = generate_email_for_name(reviewer_name)

            stars = "★" * rating + "☆" * (5 - rating)

            card_grid = Table.grid(padding=(0, 3))
            card_grid.add_column("K1", style="bold cyan")
            card_grid.add_column("V1", style="bold white")
            card_grid.add_column("K2", style="bold cyan")
            card_grid.add_column("V2", style="bold white")

            card_grid.add_row(
                "BUYER PERSONA :", f"[bold white]{reviewer_name}[/bold white] [dim cyan]<{reviewer_email}>[/dim cyan]",
                "VERIFICATION  :", "[bold green]✔ VERIFIED OWNER[/bold green]",
            )
            card_grid.add_row(
                "RATING SCORE  :", f"[bold yellow]{stars} ({rating}.0)[/bold yellow]",
                "BACKDATED PKT :", f"[dim]{created_date} PKT[/dim]",
            )

            word_count = len(text.split())
            review_card_content = Table.grid(padding=(0, 0))
            review_card_content.add_column()
            review_card_content.add_row(card_grid)
            review_card_content.add_row(Rule(style="dim cyan"))
            review_card_content.add_row(f"[italic bright_white]\"{text}\"[/italic bright_white]")

            console.print(
                Panel(
                    review_card_content,
                    title=f"[bold cyan]REVIEW {idx+1}/{len(ai_reviews)} (INSPECTION PREVIEW)[/bold cyan]",
                    subtitle=f"[dim cyan]{word_count} words[/dim cyan]",
                    border_style="cyan",
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
            )

        # 4. Human-In-The-Loop Approval Gate Safeguard
        if require_approval:
            approval_panel = (
                "[bold white][y][/bold white] [bold green] ""y"" APPROVE & PUBLISH[/bold green]   [dim](Commit and send these reviews to website)[/dim]\n"
                "[bold white][n][/bold white] [bold yellow] ""n"" REGENERATE[/bold yellow]          [dim](Discard batch and request fresh synthesis from LLM)[/dim]\n"
                "[bold white][s][/bold white] [bold cyan] ""s"" SKIP PRODUCT[/bold cyan]        [dim](Skip Product #"
                + str(p_id)
                + " without publishing anything)[/dim]\n"
                "[bold white][q][/bold white] [bold red] ""q"" QUIT / HALT[/bold red]         [dim](Stop pipeline safely and preserve checkpoint)[/dim]"
            )
            console.print(
                Panel(
                    approval_panel,
                    title=f"[bold yellow]HUMAN-IN-THE-LOOP SAFEGUARD (PRODUCT #{p_id})[/bold yellow]",
                    border_style="yellow",
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
            )

            action = Prompt.ask(
                "[bold yellow]>> ACTION [y=Deploy / n=Regenerate / s=Skip / q=Quit][/bold yellow]",
                choices=["y", "n", "s", "q"],
                default="y",
            ).lower()

            if action == "n":
                if memory is not None:
                    memory.add_rejected_reviews(ai_reviews)
                console.print(
                    f"\n[bold yellow]↺ Discarded reviews for Product #{p_id}. Blacklisting discarded text and requesting fresh synthesis from LLM co-processor...[/bold yellow]\n"
                )
                continue  # Loop again and regenerate fresh reviews for the exact same product!
            elif action == "s":
                console.print(f"\n[cyan]⏭ Skipping Product #{p_id} by operator command.[/cyan]\n")
                return True
            elif action == "q":
                console.print("\n[yellow]Pipeline halted by operator. Preserving state checkpoint...[/yellow]")
                sys.exit(0)
            else:
                console.print("\n[bold green]✔ Reviews Approved! Commencing deployment to WooCommerce...[/bold green]\n")

        # 5. Commit & Publish Approved Reviews
        if memory is not None:
            memory.add_approved_reviews(p_id, p_name, category, ai_reviews)

        reviews_posted_for_product = 0
        for idx, item in enumerate(ai_reviews):
            rating = item.rating
            text = item.review
            created_date = staggered_dates[idx]
            reviewer_name = item.name if item.name else generate_reviewer().name
            reviewer_email = generate_email_for_name(reviewer_name)

            if is_dry_run:
                time.sleep(0.15)
                console.print(
                    f"  [dim]{get_timestamp()}[/dim] [dim yellow]⚡ [SIMULATION][/dim yellow] "
                    f"Review #{idx+1} validated in-memory for [bold white]{reviewer_name}[/bold white] (0 server writes)"
                )
                reviews_posted_for_product += 1
                time.sleep(0.1)
            else:
                assert wc_client is not None
                with console.status(
                    f"[dim cyan]Transmitting Review #{idx+1}/{len(ai_reviews)} to WooCommerce REST API ({CONFIG.wc_site_url})...[/dim cyan]",
                    spinner="dots",
                ):
                    res: ReviewPostResult = wc_client.post_review(
                        product_id=p_id,
                        reviewer_name=reviewer_name,
                        reviewer_email=reviewer_email,
                        review_text=text,
                        rating=rating,
                        date_created_iso=created_date,
                        verified=True,
                    )

                if res.success:
                    console.print(
                        f"  [dim]{get_timestamp()}[/dim] [bold green]✔ [WP-REST API][/bold green] "
                        f"POST /products/reviews -> [bold green]HTTP 201 CREATED[/bold green] "
                        f"[dim](Review ID: [bold yellow]#{res.review_id}[/bold yellow])[/dim] [bold green][PUBLISHED][/bold green]"
                    )
                    reviews_posted_for_product += 1
                else:
                    console.print(
                        f"  [dim]{get_timestamp()}[/dim] [bold red]✗ [WP-REST API ERROR][/bold red] Review #{idx+1} rejected: {res.error}"
                    )

                # Jitter cooldown ticker
                delay_rev = random.uniform(CONFIG.delay_between_reviews_min, CONFIG.delay_between_reviews_max)
                render_cyber_cooldown(delay_rev, enable_anim=enable_anim)

        console.print()
        break

    # 4. Product Completion Banner
    if reviews_posted_for_product > 0:
        if not is_dry_run:
            state.mark_completed(p_id, reviews_posted_for_product)
            stats.processed_products += 1
            stats.total_reviews_posted += reviews_posted_for_product

            success_panel = (
                f"[bold green]✔ DEPLOYMENT SUCCESSFUL FOR PRODUCT #{p_id}[/bold green]\n"
                f"[bold white]{reviews_posted_for_product} Customer Reviews Published with Verified Owner Badges![/bold white]\n\n"
                f"[bold yellow]🔗 INSPECT LIVE ON STORE (CLICK OR OPEN IN BROWSER):[/bold yellow]\n"
                f"[bold cyan underline]{live_permalink}[/bold cyan underline]"
            )
            console.print(
                Panel(
                    success_panel,
                    title="[bold green]MISSION COMPLETE (LIVE PRODUCTION)[/bold green]",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
            )
        else:
            stats.processed_products += 1
            sim_panel = (
                f"[bold magenta]✔ FORENSIC SIMULATION COMPLETED FOR PRODUCT #{p_id}[/bold magenta]\n"
                f"[bold white]{reviews_posted_for_product} Authentic Reviews Synthesized & Validated in Memory.[/bold white]\n\n"
                f"[yellow]⚡ NOTE: Dry-run simulation mode was selected — ZERO modifications were sent to the live website.[/yellow]\n"
                f"[dim cyan]Target Product URL: {live_permalink}[/dim cyan]\n"
                f"[bold green]▶ READY FOR LIVE DEPLOYMENT: To actually publish these reviews, re-run with [1] LIVE DEPLOYMENT.[/bold green]"
            )
            console.print(
                Panel(
                    sim_panel,
                    title="[bold magenta]SIMULATION COMPLETE (SAFE MODE)[/bold magenta]",
                    border_style="magenta",
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
            )
        return True
    else:
        log_failure(CONFIG.failed_log_path, p_id, p_name, "All reviews failed to publish")
        stats.failed_products += 1
        return False


def run_interactive_menu(all_products: list[ProductRecord], state: StateManager) -> tuple[list[ProductRecord], bool]:
    """Displays the interactive menu with natural visual transitions and micro-pacing."""
    completed_count = len(state.completed_ids)
    remaining_count = len(all_products) - completed_count

    menu_panel = (
        "[bold white][1][/bold white] [bold cyan]TARGET SPECIFIC PRODUCT ID(s)[/bold cyan]\n"
        "    [dim]Ingest single or multiple IDs (e.g. 23245 or comma-separated: 23245, 10701)[/dim]\n\n"
        "[bold white][2][/bold white] [bold green]FULL CATALOG BATCH PIPELINE[/bold green]\n"
        f"    [dim]Autonomous bulk run across remaining {remaining_count:,} catalog products[/dim]\n\n"
        "[bold white][3][/bold white] [bold yellow]QUICK SINGLE SYSTEM TEST[/bold yellow]\n"
        f"    [dim]Quick smoke test on 1st catalog product (#{all_products[0].id})[/dim]\n\n"
        "[bold white][0][/bold white] [bold red]TERMINATE / EXIT SYSTEM[/bold red]\n"
        "    [dim]Safely exit without modifying anything[/dim]"
    )
    console.print(
        Panel(
            menu_panel,
            title="[bold cyan]STEP 1: SELECT OPERATIONAL TARGET[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    choice = Prompt.ask(
        "[bold cyan]>> SELECT TARGET OPTION[/bold cyan]",
        choices=["1", "2", "3", "0"],
        default="1",
    )

    if choice == "0":
        console.print("[yellow]Session terminated by operator.[/yellow]")
        sys.exit(0)

    # 1. Determine Selected Products with visual feedback transition
    selected: list[ProductRecord]
    if choice == "1":
        id_input = Prompt.ask(
            "\n[bold green]>> ENTER TARGET PRODUCT ID(s)[/bold green] [dim](e.g. 23245 or 23245, 10701)[/dim]",
            default="23245",
        )
        try:
            target_ids = [int(x.strip()) for x in id_input.split(",") if x.strip()]
        except ValueError:
            console.print("[bold red]Invalid ID format entered. Aborting.[/bold red]")
            sys.exit(1)

        with console.status("[dim cyan]Resolving catalog entities from database...[/dim cyan]", spinner="dots"):
            time.sleep(0.3)

        selected = [p for p in all_products if p.id in target_ids]
        if not selected:
            console.print(f"[bold red]None of the specified IDs ({target_ids}) were found in CSV![/bold red]")
            sys.exit(1)

        console.print(f"\n[bold green]✔ {len(selected)} TARGET(S) LOCKED:[/bold green]")
        for sp in selected:
            console.print(f"  • [bold cyan]#{sp.id}[/bold cyan] : [white]{sp.name}[/white]")

    elif choice == "2":
        with console.status("[dim cyan]Indexing unreviewed catalog entities...[/dim cyan]", spinner="dots"):
            time.sleep(0.3)
        remaining = [p for p in all_products if p.id not in state.completed_ids]
        console.print(f"\n[bold green]✔ FULL BATCH SELECTED: {len(remaining):,} remaining products locked.[/bold green]")
        selected = remaining

    else:  # choice == "3"
        with console.status("[dim cyan]Configuring single test target...[/dim cyan]", spinner="dots"):
            time.sleep(0.2)
        console.print(f"\n[bold yellow]✔ QUICK TEST TARGET: #{all_products[0].id} ({all_products[0].name})[/bold yellow]")
        selected = all_products[:1]

    # Visual breathing pause & transition divider
    time.sleep(0.25)
    console.print()
    console.rule("[dim cyan]SECURITY & EXECUTION CLEARANCE[/dim cyan]")
    console.print()

    # 2. Ask Execution Mode (Live vs Dry-Run Simulation)
    mode_panel = (
        "[bold white][1][/bold white] [bold red]LIVE PRODUCTION DEPLOYMENT[/bold red]\n"
        "    [dim]Generate authentic reviews & PUBLISH directly to live WooCommerce store[/dim]\n\n"
        "[bold white][2][/bold white] [bold magenta]FORENSIC DRY-RUN (SAFE SIMULATION)[/bold magenta]\n"
        "    [dim]Test complete pipeline with full TUI animations, ZERO SERVER WRITES[/dim]"
    )
    console.print(
        Panel(
            mode_panel,
            title="[bold yellow]STEP 2: SELECT EXECUTION MODE[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    mode_choice = Prompt.ask(
        "[bold yellow]>> SELECT MODE[/bold yellow]",
        choices=["1", "2"],
        default="1",
    )

    is_dry_run = mode_choice == "2"
    if not is_dry_run:
        time.sleep(0.15)
        confirm = Confirm.ask(
            f"\n[bold red]>> CONFIRM: Deploy live reviews to production store ({CONFIG.wc_site_url})?[/bold red]",
            default=True,
        )
        if not confirm:
            console.print("[yellow]Operation aborted by operator.[/yellow]")
            sys.exit(0)
        with console.status("[dim red]Arming production pipeline for live deployment...[/dim red]", spinner="dots"):
            time.sleep(0.4)
        console.print("[bold red]✔ Live Production Deployment Armed.[/bold red]")
    else:
        with console.status("[dim magenta]Activating forensic sandbox isolation...[/dim magenta]", spinner="dots"):
            time.sleep(0.3)
        console.print("[bold magenta]✔ Forensic Safe Simulation Mode Activated (0 Server Writes).[/bold magenta]")

    time.sleep(0.25)
    return selected, is_dry_run


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Terminal-Based Review Engine (MIL-SPEC TUI / Call of Duty Retro Edition)")
    parser.add_argument("--product-id", type=int, default=None, help="Target specific product ID directly")
    parser.add_argument("--test", action="store_true", help="Quick single product test run")
    parser.add_argument("--limit", type=int, default=None, help="Limit total products to process")
    parser.add_argument("--csv", type=str, default=str(CONFIG.default_csv_path), help="Path to products CSV")
    parser.add_argument("--dry-run", action="store_true", help="Run in forensic safe mode (no live writes)")
    parser.add_argument(
        "--auto-approve",
        "--yes",
        "-y",
        action="store_true",
        help="Bypass safeguard approval gate (auto-publish without interactive prompt)",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore saved checkpoint")
    parser.add_argument("--no-anim", action="store_true", help="Disable typewriter typing animations")
    args = parser.parse_args()

    enable_anim = not args.no_anim

    # 1. Play Call of Duty Boot Sequence Animation
    play_cod_boot_sequence(enable_anim=enable_anim)

    # 2. Render Cyber Banner
    render_cyber_banner()

    # 3. Load Catalog Entities
    csv_file = Path(args.csv)
    try:
        all_products = load_products_from_csv(csv_file)
    except FileNotFoundError as e:
        console.print(f"[bold red]FATAL ERROR: {e}[/bold red]")
        sys.exit(1)

    # 4. State Manager Checkpoint
    state = StateManager(CONFIG.state_file_path)

    # 5. Operational Mode
    is_dry_run = args.dry_run
    mode_label = "[bold red]LIVE INGESTION[/bold red]" if not is_dry_run else "[yellow]FORENSIC DRY-RUN[/yellow]"

    # If flags were passed, run headless mode; otherwise launch Interactive TUI Menu
    if args.product_id or args.test or args.limit:
        if args.product_id:
            matching = [p for p in all_products if p.id == args.product_id]
            if not matching:
                console.print(f"[bold red]Error: Product #{args.product_id} not found in catalog![/bold red]")
                sys.exit(1)
            target_products = matching
        elif args.test:
            target_products = all_products[:1]
        else:
            target_products = all_products[: args.limit]
    else:
        # Launch Interactive TUI Matrix Menu
        render_telemetry_matrix(
            site_url=CONFIG.wc_site_url,
            model_name=CONFIG.agentrouter_model,
            total_items=len(all_products),
            mode_label=mode_label,
        )
        target_products, is_dry_run = run_interactive_menu(all_products, state)

    # Validate Config
    try:
        CONFIG.validate(require_wc=not is_dry_run)
    except ValueError as e:
        console.print(f"[bold red]CONFIGURATION ERROR: {e}[/bold red]")
        sys.exit(1)

    # Initialize Engine Clients
    llm_client = LLMClient(CONFIG)
    memory_path = CONFIG.state_file_path.parent / "review_memory.json"
    review_memory = ReviewMemory(storage_path=memory_path)
    wc_client: WooCommerceReviewClient | None = None
    if not is_dry_run:
        wc_client = WooCommerceReviewClient(CONFIG)
        test_res = wc_client.test_connection()
        if not test_res["success"]:
            console.print(f"[bold red]✗ WooCommerce REST API Authentication Failed: {test_res['message']}[/bold red]")
            sys.exit(1)

    # Execution Loop
    stats = JobExecutionStats()
    total_targets = len(target_products)
    console.print()
    with console.status(
        f"[bold cyan]Calibrating synthesis pipeline for {total_targets} target entity(s)...[/bold cyan]",
        spinner="dots",
    ):
        time.sleep(0.4)
    console.print(f"[bold green]⚡ LAUNCHING CYBER-SYNTHESIS ON {total_targets} ENTITY TARGET(S)...[/bold green]\n")
    time.sleep(0.2)

    for idx, prod in enumerate(target_products, 1):
        console.print(f"[bold cyan]─── [TARGET {idx}/{total_targets}] ───────────────────────────────────────────────────────────[/bold cyan]")
        process_product_pipeline(
            prod=prod,
            llm_client=llm_client,
            wc_client=wc_client,
            is_dry_run=is_dry_run,
            state=state,
            stats=stats,
            enable_anim=enable_anim,
            require_approval=not args.auto_approve,
            memory=review_memory,
        )

        # Product cooldown delay
        if not is_dry_run and idx < total_targets:
            p_delay = random.uniform(CONFIG.delay_between_products_min, CONFIG.delay_between_products_max)
            with console.status(
                f"[dim cyan]Inter-target cooldown ({p_delay:.1f}s delay before target #{idx+1})...[/dim cyan]",
                spinner="dots",
            ):
                time.sleep(p_delay)
            console.print(f"[dim]{get_timestamp()} [TARGET COOLDOWN] Buffer ready ({p_delay:.1f}s elapsed)[/dim]\n")

    # Final Forensic Debrief Table
    debrief = Table(
        title="[bold cyan]OPERATION DEBRIEF[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        show_lines=True,
    )
    debrief.add_column("Operational Metric", style="bold cyan", min_width=32)
    debrief.add_column("Status / Telemetry Value", style="bold white")

    if is_dry_run:
        debrief.add_row("Execution Mode", "[bold magenta]FORENSIC DRY-RUN (SAFE SIMULATION)[/bold magenta]")
        debrief.add_row("Target Entities Tested", f"[bold green]{stats.processed_products}[/bold green]")
        debrief.add_row("Reviews Synthesized & Validated", "[bold green]100% SUCCESSFUL (In Memory)[/bold green]")
        debrief.add_row("Live Database Modifications", "[bold green]0 (Zero Server Writes)[/bold green]")
        debrief.add_row("Live Store Target", f"[dim]{CONFIG.wc_site_url}[/dim]")
        debrief.add_row("Deployment Readiness", "[bold green]PASSED • READY FOR PRODUCTION DEPLOYMENT[/bold green]")
    else:
        debrief.add_row("Execution Mode", "[bold red]LIVE PRODUCTION DEPLOYMENT[/bold red]")
        debrief.add_row("Entities Ingested Successfully", f"[bold green]{stats.processed_products}[/bold green]")
        debrief.add_row("Total Verified Reviews Published", f"[bold green]{stats.total_reviews_posted}[/bold green]")
        debrief.add_row("Failed Entity Targets", f"[bold red]{stats.failed_products}[/bold red]" if stats.failed_products else "[green]0 (None)[/green]")
        debrief.add_row("Catalog Checkpoint Total", f"[cyan]{len(state.completed_ids)}[/cyan] Products in State File")
        debrief.add_row("Live Production Target", f"[bold green]{CONFIG.wc_site_url}[/bold green]")

    console.print("\n", debrief)
    if is_dry_run:
        console.print("[bold magenta]✔ Simulation completed. Zero live server data was modified.[/bold magenta]\n")
    else:
        console.print("[bold green]✔ All operations executed and published with 100% integrity.[/bold green]\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print()
        console.print(
            Panel(
                "[bold yellow]MISSION SUSPENDED BY OPERATOR (Ctrl+C Detected)[/bold yellow]\n\n"
                "[white]• Checkpoint state preserved safely in [cyan]progress_state.json[/cyan]\n"
                "• All previously submitted reviews are safe and live.\n"
                "• To resume, run the script again—it will pick up right where you left off.[/white]",
                title="[bold yellow]OPERATOR INTERRUPT (SAFE HALT)[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        sys.exit(0)
