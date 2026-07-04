"""
Support Triage Dashboard — batch inbox view for the multi-stage RAG agent.

Runs a queue of real support tickets through the classify → retrieve → respond →
safety-gate pipeline, shows how each was routed, and lets you drill into any
ticket to see *why* it was auto-answered or escalated.

Run:  streamlit run app/dashboard.py
Works with no API key (extractive/offline mode). Set ANTHROPIC_API_KEY in a
.env file at the repo root for full Claude generation.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from pipeline_runner import analyze_all, llm_mode, load_index, load_tickets

st.set_page_config(page_title="Support Triage Dashboard", page_icon="🛎️", layout="wide")

# ---------------------------------------------------------------- styling ----
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1250px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .hero {display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap;}
      .hero h1 {margin:0; font-size:1.9rem;}
      .hero .sub {color:#64748b; font-size:1rem;}
      .pill {display:inline-block; padding:.16rem .6rem; border-radius:999px;
             font-size:.78rem; font-weight:600; line-height:1.4;}
      .pill-replied   {background:#dcfce7; color:#166534;}
      .pill-escalated {background:#fef3c7; color:#92400e;}
      .pill-mode      {background:#e0e7ff; color:#3730a3;}
      .pill-live      {background:#dcfce7; color:#166534;}
      .card {border:1px solid #e5e7eb; border-radius:14px; padding:1rem 1.15rem;
             background:#ffffff;}
      .stage {border-left:3px solid #6366f1; padding:.15rem 0 .15rem .8rem;
              margin:.35rem 0;}
      .stage b {color:#111827;}
      .stage .d {color:#374151;}
      .stage .n {color:#6b7280; font-size:.85rem; font-style:italic;}
      .src {border:1px solid #e5e7eb; border-radius:10px; padding:.55rem .8rem;
            margin:.35rem 0; background:#f8fafc;}
      .src .t {font-weight:600; color:#1e293b;}
      .src .m {color:#64748b; font-size:.8rem;}
      [data-testid="stMetricValue"] {font-size:1.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------- data loading --
@st.cache_resource(show_spinner="Indexing 774 support documents…")
def _index():
    return load_index()


@st.cache_data(show_spinner="Triaging ticket queue…")
def _results():
    idx = _index()
    tickets = load_tickets()
    results = analyze_all(tickets, idx)
    # Return plain dicts so Streamlit can cache/serialize cleanly.
    rows = []
    for i, r in enumerate(results):
        rows.append(
            {
                "idx": i,
                "Subject": r.subject,
                "Company": r.company,
                "Type": r.request_type,
                "Product area": r.product_area,
                "Status": r.status,
                "Safety-gate acted": "yes" if (r.was_revised or r.status == "escalated") else "—",
                "issue": r.ticket.issue,
                "reasoning": r.classification.reasoning,
                "justification": r.justification,
                "response": r.final_response,
                "revision_notes": r.revision_notes,
                "stages": r.stages,
                "sources": [
                    {"title": c.title, "domain": c.domain, "score": round(c.score, 2)}
                    for c in r.chunks
                ],
            }
        )
    stats = idx.stats
    return rows, {"files": stats.files, "chunks": stats.chunks, "domains": stats.domains}


rows, corpus = _results()
df = pd.DataFrame(rows)
mode = llm_mode()

# --------------------------------------------------------------------- hero --
mode_pill = (
    '<span class="pill pill-live">● Claude live</span>'
    if mode == "claude"
    else '<span class="pill pill-mode">offline / extractive mode</span>'
)
st.markdown(
    f"""<div class="hero">
      <h1>🛎️ Support Triage Dashboard</h1>
      <span class="sub">Multi-stage RAG agent · auto-answer vs. escalate</span>
      {mode_pill}
    </div>""",
    unsafe_allow_html=True,
)
st.markdown(
    f'<span class="sub">Knowledge base: <b>{corpus["files"]}</b> documents · '
    f'<b>{corpus["chunks"]:,}</b> retrievable chunks · '
    f'domains: {", ".join(corpus["domains"])}</span>',
    unsafe_allow_html=True,
)
st.write("")

# ------------------------------------------------------------------- KPIs -----
total = len(df)
replied = int((df["Status"] == "replied").sum())
escalated = int((df["Status"] == "escalated").sum())
gate_acted = int((df["Safety-gate acted"] == "yes").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Tickets triaged", total)
k2.metric("Auto-resolved", f"{replied}", f"{replied/total:.0%} of queue")
k3.metric("Escalated to human", f"{escalated}", f"{escalated/total:.0%} of queue", delta_color="off")
k4.metric("Safety-gate actions", gate_acted, "escalated or revised", delta_color="off")

st.write("")

# ------------------------------------------------------------------ charts ----
c1, c2 = st.columns([1, 1.3])
with c1:
    st.markdown("**Routing outcome**")
    st.bar_chart(
        pd.DataFrame(
            {"count": [replied, escalated]}, index=["auto-resolved", "escalated"]
        ),
        color="#6366f1",
        horizontal=True,
    )
with c2:
    st.markdown("**Volume by product**")
    by_company = df.groupby("Company").size().sort_values(ascending=False)
    st.bar_chart(by_company, color="#818cf8")

st.divider()

# ---------------------------------------------------------------- inbox table -
left, right = st.columns([1.15, 1])

with left:
    st.markdown("### 📥 Ticket queue")
    f1, f2 = st.columns(2)
    company_opts = ["All"] + sorted(df["Company"].unique())
    status_opts = ["All", "replied", "escalated"]
    company_sel = f1.selectbox("Company", company_opts)
    status_sel = f2.selectbox("Status", status_opts)

    view = df.copy()
    if company_sel != "All":
        view = view[view["Company"] == company_sel]
    if status_sel != "All":
        view = view[view["Status"] == status_sel]

    table = view[["Subject", "Company", "Type", "Product area", "Status", "Safety-gate acted"]]
    event = st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Status": st.column_config.TextColumn(width="small"),
            "Safety-gate acted": st.column_config.TextColumn("Gate", width="small"),
        },
    )

    # Resolve selection back to the master dataframe index.
    sel_rows = event.selection.rows if event and event.selection else []
    selected_idx = int(view.iloc[sel_rows[0]]["idx"]) if sel_rows else int(df.iloc[0]["idx"])

# --------------------------------------------------------------- drill-in -----
with right:
    r = df[df["idx"] == selected_idx].iloc[0]
    pill = f'pill-{r["Status"]}'
    st.markdown("### 🔎 Ticket detail")
    st.markdown(
        f'**{r["Subject"]}** &nbsp; <span class="pill {pill}">{r["Status"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="src m">{r["Company"]} · {r["Type"]} · {r["Product area"]}</span>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.caption("Incoming ticket")
        st.write(r["issue"])

    st.markdown("**Pipeline trace**")
    for s in r["stages"]:
        note = f'<div class="n">{s["note"]}</div>' if s.get("note") else ""
        st.markdown(
            f'<div class="stage"><b>{s["name"]}</b> — <span class="d">{s["detail"]}</span>{note}</div>',
            unsafe_allow_html=True,
        )

    if r["sources"]:
        with st.expander(f"📚 Retrieved sources ({len(r['sources'])})", expanded=False):
            for s in r["sources"]:
                st.markdown(
                    f'<div class="src"><span class="t">{s["title"]}</span>'
                    f'<span class="m"> · {s["domain"]} · BM25 {s["score"]}</span></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No retrieval — ticket was escalated before hitting the knowledge base.")

    st.markdown("**Agent decision**")
    if r["Status"] == "escalated":
        st.warning(r["justification"])
    else:
        st.success(r["justification"])

    st.markdown("**Response sent to customer**")
    with st.container(border=True):
        st.write(r["response"])

st.divider()
st.caption(
    "Built by Viet Tran · classify → retrieve (BM25) → ground → safety-gate. "
    "Every automated answer passes a second-model review before it reaches a customer."
)
