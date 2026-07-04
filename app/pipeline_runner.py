"""Rich wrapper around the support-triage pipeline.

Re-runs the same classify -> retrieve -> respond -> safety-review stages used by
`code/main.py`, but captures every intermediate artifact so the dashboard can
show *why* each ticket was answered or escalated.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make the existing `code/` modules importable without changing them.
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from classifier import classify_ticket  # noqa: E402
from indexer import CorpusIndex, build_corpus_index  # noqa: E402
from llm import has_api_key  # noqa: E402
from models import Classification, RetrievedChunk, Ticket  # noqa: E402
from responder import generate_response, retrieve  # noqa: E402
from safety_gate import review_response  # noqa: E402
from utils import escalation_response, invalid_response, read_tickets  # noqa: E402


@dataclass
class TicketResult:
    ticket: Ticket
    classification: Classification
    chunks: list[RetrievedChunk]
    draft_response: str
    final_response: str
    status: str
    justification: str
    revision_notes: str
    escalated_before_retrieval: bool = False
    stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def company(self) -> str:
        return self.classification.company

    @property
    def product_area(self) -> str:
        return self.classification.product_area or "—"

    @property
    def request_type(self) -> str:
        return self.classification.request_type

    @property
    def subject(self) -> str:
        return self.ticket.subject or "(no subject)"

    @property
    def was_revised(self) -> bool:
        note = (self.revision_notes or "").strip().lower()
        return note not in ("", "no changes")


def analyze_ticket(ticket: Ticket, index: CorpusIndex) -> TicketResult:
    """Mirror of main.process_ticket, but keeps every intermediate artifact."""
    stages: list[dict[str, Any]] = []
    classification = classify_ticket(ticket)
    stages.append(
        {
            "name": "1 · Classify",
            "detail": f"{classification.company} / {classification.request_type} / "
            f"{classification.product_area or '—'} → hint: {classification.status_hint}",
            "note": classification.reasoning,
        }
    )

    # Invalid tickets: canned reply, no retrieval.
    if classification.request_type == "invalid":
        response, justification = invalid_response(ticket)
        stages.append({"name": "2 · Retrieve", "detail": "skipped (invalid ticket)", "note": ""})
        stages.append({"name": "3 · Respond", "detail": "canned out-of-scope reply", "note": ""})
        stages.append({"name": "4 · Safety gate", "detail": "not required", "note": ""})
        return TicketResult(
            ticket, classification, [], response, response, "replied",
            justification, "no changes", escalated_before_retrieval=True, stages=stages,
        )

    # Deterministic pre-retrieval escalation (fraud, billing, prompt-injection...).
    if classification.status_hint == "escalated":
        response, _ = escalation_response(classification.reasoning)
        justification = (classification.reasoning or "Requires specialist support.").strip()
        stages.append({"name": "2 · Retrieve", "detail": "skipped — escalated pre-retrieval", "note": ""})
        stages.append({"name": "3 · Respond", "detail": "escalation message", "note": ""})
        stages.append(
            {"name": "4 · Safety gate", "detail": "escalated by classifier guardrail", "note": classification.reasoning}
        )
        return TicketResult(
            ticket, classification, [], response, response, "escalated",
            justification, "no changes", escalated_before_retrieval=True, stages=stages,
        )

    # Normal path: retrieve -> respond -> safety review.
    chunks = retrieve(index, ticket, classification)
    stages.append(
        {
            "name": "2 · Retrieve",
            "detail": f"BM25 top-{len(chunks)} from '{classification.company}' domain",
            "note": ", ".join(c.title for c in chunks[:3]),
        }
    )

    draft = generate_response(ticket, classification, chunks)
    stages.append({"name": "3 · Respond", "detail": "grounded draft generated", "note": ""})

    safety = review_response(ticket, classification, draft)
    stages.append(
        {
            "name": "4 · Safety gate",
            "detail": f"verdict: {safety.final_status}"
            + (" (revised)" if safety.revision_notes not in ("", "no changes") else ""),
            "note": safety.revision_notes,
        }
    )

    if safety.final_status == "escalated":
        justification = (safety.revision_notes or classification.reasoning or "Escalated on safety review.").strip()
    else:
        top = chunks[0].title if chunks else "support documentation"
        justification = f"Answered from '{top}'. {classification.reasoning}".strip()

    return TicketResult(
        ticket=ticket,
        classification=classification,
        chunks=chunks,
        draft_response=draft,
        final_response=safety.revised_response,
        status=safety.final_status,
        justification=justification,
        revision_notes=safety.revision_notes,
        stages=stages,
    )


def load_index(data_dir: Path | None = None) -> CorpusIndex:
    return build_corpus_index(data_dir or (REPO_ROOT / "data"))


def load_tickets(csv_path: Path | None = None) -> list[Ticket]:
    return read_tickets(csv_path or (REPO_ROOT / "support_tickets" / "support_tickets.csv"))


def analyze_all(tickets: list[Ticket], index: CorpusIndex) -> list[TicketResult]:
    return [analyze_ticket(t, index) for t in tickets]


def llm_mode() -> str:
    return "claude" if has_api_key() else "offline"
