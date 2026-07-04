# Support Triage Dashboard

Batch inbox view for the multi-stage RAG support agent. Runs a queue of real
support tickets through the **classify → retrieve (BM25) → ground → safety-gate**
pipeline and shows how each ticket was routed, plus a drill-in explaining *why*
each was auto-answered or escalated.

## Run locally
```bash
pip install -r app/requirements.txt
streamlit run app/dashboard.py
```

Runs with **no API key** (offline/extractive mode). For full Claude generation,
add a `.env` file at the repo root:
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Deploy a public demo (free)
Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → point it at
`app/dashboard.py`. No key needed for the public demo; it falls back to the
extractive responder.

## What it shows
- **KPIs:** auto-resolution rate, escalation rate, safety-gate actions.
- **Routed queue:** every ticket with company, product area, and status.
- **Drill-in:** the 4-stage pipeline trace, retrieved sources with BM25 scores,
  the safety-gate verdict, and the final customer response.

The safety gate is the point: every automated answer passes a second-model
review before it reaches a customer, and billing/fraud/prompt-injection tickets
are escalated instead of answered.
