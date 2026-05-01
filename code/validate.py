from __future__ import annotations

import csv
from pathlib import Path

from indexer import build_corpus_index
from main import process_ticket
from models import Ticket
from utils import load_environment, repo_root


def normalize(value: object) -> str:
    return ("" if value is None else str(value)).strip().lower().replace("-", "_").replace(" ", "_")


def read_sample(path: Path) -> list[tuple[Ticket, dict[str, str]]]:
    rows: list[tuple[Ticket, dict[str, str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            ticket = Ticket(
                issue=row.get("Issue") or "",
                subject=row.get("Subject") or "",
                company=row.get("Company") or "None",
                row_number=index,
            )
            expected = {
                "status": normalize(row.get("Status")),
                "request_type": normalize(row.get("Request Type")),
                "product_area": normalize(row.get("Product Area")),
            }
            rows.append((ticket, expected))
    return rows


def main() -> None:
    load_environment()
    root = repo_root()
    sample_path = root / "support_tickets" / "sample_support_tickets.csv"
    index = build_corpus_index(root / "data")
    sample_rows = read_sample(sample_path)

    totals = {"status": 0, "request_type": 0, "product_area": 0}
    mismatches: list[dict[str, str]] = []
    for ticket, expected in sample_rows:
        actual = process_ticket(ticket, index)
        actual_values = {
            "status": normalize(actual.status),
            "request_type": normalize(actual.request_type),
            "product_area": normalize(actual.product_area),
        }
        for field in totals:
            if actual_values[field] == expected[field]:
                totals[field] += 1
            else:
                mismatches.append(
                    {
                        "row": str(ticket.row_number),
                        "field": field,
                        "expected": expected[field],
                        "actual": actual_values[field],
                        "subject": ticket.subject,
                    }
                )

    count = len(sample_rows)
    print("Sample validation")
    print("=================")
    for field, correct in totals.items():
        print(f"{field}: {correct}/{count} ({correct / count:.0%})")

    if mismatches:
        print("\nMismatches")
        print("----------")
        for mismatch in mismatches:
            print(
                f"row {mismatch['row']} {mismatch['field']}: "
                f"expected={mismatch['expected']!r} actual={mismatch['actual']!r} "
                f"subject={mismatch['subject']!r}"
            )
    else:
        print("\nNo mismatches.")

    if totals["status"] < count or totals["request_type"] < count or totals["product_area"] < 8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
