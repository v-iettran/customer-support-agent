from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from classifier import classify_ticket
from indexer import CorpusIndex, build_corpus_index
from models import OutputRow, Ticket
from responder import retrieve_and_respond
from safety_gate import review_response
from utils import (
    escalation_response,
    invalid_response,
    load_environment,
    make_output_row,
    read_tickets,
    repo_root,
    write_output,
)


def process_ticket(ticket: Ticket, index: CorpusIndex) -> OutputRow:
    classification = classify_ticket(ticket)

    if classification.request_type == "invalid":
        response, justification = invalid_response(ticket)
        return make_output_row(ticket, classification, response, "replied", justification)

    if classification.status_hint == "escalated":
        response, justification = escalation_response(classification.reasoning)
        return make_output_row(ticket, classification, response, "escalated", justification)

    draft_response, _chunks = retrieve_and_respond(index, ticket, classification)
    safety = review_response(ticket, classification, draft_response)
    if safety.final_status == "escalated":
        justification = f"Escalated after safety review: {safety.revision_notes}."
    else:
        justification = classification.reasoning or "Answered using relevant support documentation from the local corpus."
    return make_output_row(ticket, classification, safety.revised_response, safety.final_status, justification)


def run(input_path: Path, output_path: Path, data_dir: Path) -> list[OutputRow]:
    load_environment()
    print(f"Building corpus index from {data_dir}...")
    index = build_corpus_index(data_dir)
    stats = index.stats
    print(f"Indexed {stats.files} files into {stats.chunks} chunks: {stats.domains}")

    tickets = read_tickets(input_path)
    print(f"Processing {len(tickets)} tickets from {input_path}...")
    rows: list[OutputRow] = []
    for ticket in tickets:
        label = f"#{ticket.row_number}" if ticket.row_number else ""
        print(f"  {label} {ticket.company}: {ticket.subject or ticket.issue[:50]}")
        rows.append(process_ticket(ticket, index))

    write_output(rows, output_path)
    counts = Counter(row.status for row in rows)
    types = Counter(row.request_type for row in rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Status counts: {dict(counts)}")
    print(f"Request type counts: {dict(types)}")
    return rows


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run the HackerRank Orchestrate support triage agent.")
    parser.add_argument("--input", type=Path, default=root / "support_tickets" / "support_tickets.csv")
    parser.add_argument("--output", type=Path, default=root / "support_tickets" / "output.csv")
    parser.add_argument("--data", type=Path, default=root / "data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input, args.output, args.data)


if __name__ == "__main__":
    main()
