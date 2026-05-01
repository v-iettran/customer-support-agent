# Support Triage Agent

A terminal-based support agent for HackerRank Orchestrate. It classifies and responds to tickets across HackerRank, Claude, and Visa using the local `data/` support corpus and writes predictions to `support_tickets/output.csv`.

## Architecture

The agent uses a three-stage pipeline:

1. **Classify & Route**: Determines company, request type, product area, escalation status, and retrieval query.
2. **Retrieve & Respond**: Builds a BM25 index over the markdown corpus, retrieves the top passages for the classified domain, and generates a grounded response.
3. **Safety Gate**: Reviews the draft response and flips high-risk, unsafe, unsupported, or adversarial tickets to escalation.

Claude API calls use `claude-sonnet-4-20250514`, temperature `0`, and a local SQLite cache at `code/.cache/llm.sqlite`. If `ANTHROPIC_API_KEY` is not present, the agent uses deterministic fallback rules and extractive corpus responses so validation and smoke tests can still run locally.

## Setup

```bash
cd code
python3 -m pip install -r requirements.txt
cd ..
cp .env.example .env
```

Add your Anthropic key to `.env`:

```bash
ANTHROPIC_API_KEY=your-key-here
```

## Run

From the repository root:

```bash
python3 code/main.py
```

This reads `support_tickets/support_tickets.csv` and writes `support_tickets/output.csv`.

Optional paths:

```bash
python3 code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv --data data
```

## Validate

```bash
python3 code/validate.py
```

Latest local validation result:

```text
status: 10/10 (100%)
request_type: 10/10 (100%)
product_area: 10/10 (100%)
```

Smoke checks after generating `output.csv`:

```text
rows=29
status_counts={'replied': 15, 'escalated': 14}
request_type_counts={'product_issue': 23, 'feature_request': 0, 'bug': 5, 'invalid': 1}
SMOKE CHECK PASSED
```

## Design Rationale

- **BM25 retrieval**: The corpus is small and keyword-heavy, so BM25 is deterministic, fast, and avoids a dependency on embedding APIs or vector stores.
- **Domain-filtered search**: Once a ticket is classified, retrieval is limited to the matching product corpus to reduce cross-domain noise.
- **Separate safety gate**: Escalation logic is explicit and independently checks for billing disputes, account actions, fraud, security disclosures, prompt injection, outages, and unsupported answers.
- **SQLite response cache**: Claude responses are cached by stage, model, prompt, and input hash to make repeated development runs deterministic and inexpensive.
- **Fallback path**: The primary design uses Claude, but local fallback logic keeps the project runnable when an API key is unavailable.
