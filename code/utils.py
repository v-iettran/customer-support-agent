from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from models import Classification, OutputRow, Ticket


OUTPUT_FIELDS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_environment() -> None:
    load_dotenv(repo_root() / ".env")


def read_tickets(path: Path) -> list[Ticket]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        tickets: list[Ticket] = []
        for index, row in enumerate(reader, start=1):
            tickets.append(
                Ticket(
                    issue=row.get("Issue") or row.get("issue") or "",
                    subject=row.get("Subject") or row.get("subject") or "",
                    company=row.get("Company") or row.get("company") or "None",
                    row_number=index,
                )
            )
    return tickets


def write_output(rows: Iterable[OutputRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def invalid_response(ticket: Ticket) -> tuple[str, str]:
    text = f"{ticket.issue} {ticket.subject}".lower()
    if any(word in text for word in ("thank you", "thanks", "thankyou")):
        return "You're welcome! Let us know if you need anything else.", "No action required; the user is expressing gratitude."
    return (
        "This request is outside the scope of our support services. We handle support for HackerRank, Claude, and Visa products only.",
        "The ticket is unrelated to any supported product domain.",
    )


def escalation_response(reason: str | None = None) -> tuple[str, str]:
    reason_text = reason.strip() if reason else "this issue requires specialist support"
    return (
        "This issue requires assistance from our support team. I've escalated your ticket so a specialist can help you directly.",
        f"Escalated because {reason_text}.",
    )


def make_output_row(
    ticket: Ticket,
    classification: Classification,
    response: str,
    status: str,
    justification: str,
) -> OutputRow:
    return OutputRow(
        issue=ticket.issue,
        subject=ticket.subject,
        company=ticket.company,
        response=response.strip(),
        product_area=classification.product_area,
        status=status,
        request_type=classification.request_type,
        justification=justification.strip(),
    )


def append_turn_log(title: str, prompt: str, summary: str, actions: list[str]) -> None:
    """Append a short project transcript entry without logging secrets."""
    root = repo_root()
    log_path = Path.home() / "hackerrank_orchestrate" / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    branch = os.environ.get("GIT_BRANCH", "unknown")
    try:
        import subprocess

        branch = (
            subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip()
            or "unknown"
        )
    except Exception:
        pass
    now = datetime.now().astimezone().isoformat()
    action_lines = "\n".join(f"* {action}" for action in actions)
    entry = f"""## [{now}] {title[:80]}

User Prompt (verbatim, secrets redacted):
{prompt}

Agent Response Summary:
{summary}

Actions:
{action_lines}

Context:
tool=GPT-5.5
branch={branch}
repo_root={root}
worktree=main
parent_agent=none

"""
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(entry)
