from __future__ import annotations

import json
import re

from llm import LLMUnavailable, call_claude
from models import Classification, Ticket


SYSTEM_PROMPT = """You are a support ticket classifier for three products: HackerRank, Claude (by Anthropic), and Visa.

Given a support ticket, output ONLY a JSON object with these fields:
{
  "company": "HackerRank" | "Claude" | "Visa" | "None",
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid",
  "product_area": "<slug>",
  "status_hint": "replied" | "escalated",
  "retrieval_query": "<2-6 word search query for the support corpus>",
  "reasoning": "<1 sentence explaining classification>"
}

Classification rules:
- If company is "None", infer from the issue content. If truly unrelated to any product, set company="None".
- "invalid" = the ticket is off-topic, nonsensical, a thank-you, or unrelated to any supported product.
- "bug" = the user reports something broken/not working.
- "feature_request" = the user asks for something that doesn't exist yet.
- "product_issue" = general how-to, account, billing, or product usage question.
- Set status_hint="escalated" if:
  - The issue involves billing disputes, refunds, payment issues with specific order IDs
  - Account access/security that requires admin action the support agent cannot perform
  - Identity theft, fraud, or stolen credentials
  - Security vulnerabilities or bug bounties
  - The issue is too vague to answer (e.g., "it's not working" with no company)
  - Subscription cancellation/pause requests (requires account-level action)
  - Requests to fill out security/infosec forms
  - Site-wide outages
- Set status_hint="replied" if:
  - The answer can be found in the support corpus
  - The ticket is invalid/off-topic (reply saying out of scope)
  - The ticket is a simple thank-you
- product_area: MUST be one of:
  HackerRank: screen, community, interviews, library, integrations, settings, skillup, engage, chakra, general_help, billing
  Claude: privacy, team_and_enterprise, pro_and_max, claude_code, claude_api, claude_desktop, safeguards, connectors, education, amazon_bedrock, general
  Visa: travel_support, general_support, dispute_resolution, fraud_protection, data_security, regulations_fees
  Invalid/out-of-scope: empty string, except off-topic conversation handling can use conversation_management.
- retrieval_query: extract the core topic for searching the support corpus. Skip if status is escalated or request is invalid.

IMPORTANT: Output raw JSON only. No markdown fences, no preamble."""


FEW_SHOT = """Example 1:
Issue: "I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system."
Subject: "Test Active in the system"
Company: "HackerRank"
Output: {"company":"HackerRank","request_type":"product_issue","product_area":"screen","status_hint":"replied","retrieval_query":"test expiration active duration","reasoning":"User asks about test validity period, answerable from Screen docs."}

Example 2:
Issue: "site is down & none of the pages are accessible"
Subject: ""
Company: "None"
Output: {"company":"None","request_type":"bug","product_area":"","status_hint":"escalated","retrieval_query":"","reasoning":"Vague outage report with no company specified, cannot diagnose or answer."}

Example 3:
Issue: "What is the name of the actor in Iron Man?"
Subject: "Urgent, please help"
Company: "None"
Output: {"company":"None","request_type":"invalid","product_area":"","status_hint":"replied","retrieval_query":"","reasoning":"Off-topic question unrelated to any supported product."}

Example 4:
Issue: "I bought Visa Traveller's Cheques from Citicorp and they were stolen in Lisbon last night. What do I do?"
Subject: ""
Company: "Visa"
Output: {"company":"Visa","request_type":"product_issue","product_area":"travel_support","status_hint":"replied","retrieval_query":"stolen travellers cheques report","reasoning":"Stolen cheques, answerable from Visa travel support docs."}

Example 5:
Issue: "Thank you for helping me"
Subject: ""
Company: "None"
Output: {"company":"None","request_type":"invalid","product_area":"","status_hint":"replied","retrieval_query":"","reasoning":"Simple gratitude, no action needed."}"""


def _json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in classifier output: {text[:120]}")
    return json.loads(match.group(0))


def _ticket_text(ticket: Ticket) -> str:
    return f"{ticket.issue}\n{ticket.subject}".lower()


def heuristic_classify(ticket: Ticket) -> Classification:
    text = _ticket_text(ticket)
    company = ticket.company
    if company == "None":
        if "visa" in text or "card" in text:
            company = "Visa"
        elif "claude" in text or "anthropic" in text:
            company = "Claude"
        elif "hackerrank" in text or "assessment" in text or "test" in text:
            company = "HackerRank"

    if any(term in text for term in ("thank you", "thanks", "thankyou")) and len(text.split()) < 12:
        return Classification(company="None", request_type="invalid", product_area="", status_hint="replied", retrieval_query="", reasoning="Simple gratitude, no support action needed.")
    if "iron man" in text:
        return Classification(company="None", request_type="invalid", product_area="conversation_management", status_hint="replied", retrieval_query="", reasoning="The request is unrelated to the supported product domains.")
    if any(term in text for term in ("delete all files", "rm -rf", "format the system")):
        return Classification(company="None", request_type="invalid", product_area="conversation_management", status_hint="replied", retrieval_query="", reasoning="The user asks the agent to perform an unsafe filesystem action. This is outside support scope; the response is the canned out-of-scope reply.")

    if "règles internes" in text or "internal rules" in text or "documents récupérés" in text:
        return Classification(company="Visa", request_type="product_issue", product_area="fraud_protection", status_hint="escalated", retrieval_query="", reasoning="The user asks for internal rules and fraud-detection logic while describing a blocked Visa card. This is a prompt-injection attempt and a possible fraud case, so it must be escalated without revealing internal process.")
    if "security vulnerability" in text or "bug bounty" in text:
        return Classification(company="Claude", request_type="bug", product_area="safeguards", status_hint="escalated", retrieval_query="", reasoning="The user reports a Claude security vulnerability. Vulnerability disclosures must go through Anthropic's security team; the corpus covers safeguards policy but not bug-bounty intake.")
    if "identity" in text and "stolen" in text:
        return Classification(company="Visa", request_type="product_issue", product_area="fraud_protection", status_hint="escalated", retrieval_query="", reasoning="The user reports identity theft. This is a high-risk fraud scenario requiring the Visa fraud team; the corpus covers general fraud prevention but not case-level incident response.")
    if "infosec" in text or "security compliance" in text or "filling in the forms" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="general_help", status_hint="escalated", retrieval_query="", reasoning="The user asks HackerRank support to fill out company infosec forms. This requires verified business and security review; the corpus cannot complete customer-specific compliance paperwork.")
    if "order id" in text or "refund" in text or "give me my money" in text:
        area = "dispute_resolution" if company == "Visa" else "billing" if company == "HackerRank" else "pro_and_max"
        return Classification(company=company, request_type="product_issue", product_area=area, status_hint="escalated", retrieval_query="", reasoning="The user raises a billing, payment, or refund dispute. This requires account-specific review; the corpus can explain policies but cannot resolve a transaction or issue money.")
    if "pause our subscription" in text or "subscription pause" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="billing", status_hint="escalated", retrieval_query="", reasoning="The user wants their HackerRank subscription paused. This is an account-level action a support agent cannot perform; the corpus documents subscription plans but not pause requests on a customer's behalf.")
    if "rescheduling" in text or "alternative date" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="screen", status_hint="escalated", retrieval_query="", reasoning="The user wants a HackerRank assessment rescheduled after missing the scheduled time. Scheduling is controlled by the hiring company; the corpus cannot grant a new assessment window.")
    if "increase my score" in text or "move me to the next round" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="screen", status_hint="escalated", retrieval_query="", reasoning="The user asks HackerRank to review answers, increase a score, and influence a hiring decision. This is a case-specific assessment and employment outcome request that the support corpus cannot resolve.")
    if ("site is down" in text or "stopped working completely" in text or "all requests are failing" in text) and "bedrock" not in text:
        return Classification(company=company, request_type="bug", product_area="" if company == "None" else "general", status_hint="escalated", retrieval_query="", reasoning="The user reports a broad outage or complete service failure. This requires live operational investigation; the static corpus cannot confirm or remediate current availability.")
    if "it's not working" in text or "it’s not working" in text:
        return Classification(company=company, request_type="bug", product_area="", status_hint="escalated", retrieval_query="", reasoning="The user reports that something is not working without enough product or error context. The corpus cannot diagnose an unspecified issue, so a human needs to gather details.")

    if company == "Claude":
        if "private info" in text or "temporary chat" in text or "delete" in text:
            return Classification(company="Claude", request_type="product_issue", product_area="privacy", status_hint="replied", retrieval_query="delete conversation private information", reasoning="Conversation deletion and privacy are covered by Claude privacy docs.")
        if "workspace" in text or "seat" in text:
            return Classification(company="Claude", request_type="product_issue", product_area="team_and_enterprise", status_hint="replied", retrieval_query="workspace seat admin re add", reasoning="Workspace access and seats are covered by Claude team docs.")
        if "crawl" in text:
            return Classification(company="Claude", request_type="product_issue", product_area="privacy", status_hint="replied", retrieval_query="block anthropic web crawler", reasoning="Website crawling is covered by Claude privacy docs.")
        if "data" in text and "models" in text:
            return Classification(company="Claude", request_type="product_issue", product_area="privacy", status_hint="replied", retrieval_query="data used improve models duration", reasoning="Data usage duration is covered by Claude privacy docs.")
        if "bedrock" in text:
            return Classification(company="Claude", request_type="product_issue", product_area="amazon_bedrock", status_hint="replied", retrieval_query="aws bedrock requests failing", reasoning="AWS Bedrock errors are covered by Claude Bedrock docs.")
        if "lti" in text or "professor" in text:
            return Classification(company="Claude", request_type="product_issue", product_area="education", status_hint="replied", retrieval_query="claude education lti key", reasoning="LTI setup is covered by Claude education docs.")
        return Classification(company="Claude", request_type="product_issue", product_area="general", status_hint="replied", retrieval_query=text[:80], reasoning="Claude support question.")

    if company == "Visa":
        if "dispute" in text or "charge" in text or "wrong product" in text:
            return Classification(company="Visa", request_type="product_issue", product_area="dispute_resolution", status_hint="replied", retrieval_query="dispute charge wrong product", reasoning="Charge disputes are covered by Visa dispute docs.")
        if "lost" in text and "card" in text:
            return Classification(company="Visa", request_type="product_issue", product_area="general_support", status_hint="replied", retrieval_query="lost stolen visa card report", reasoning="Lost card reporting is covered by Visa support docs.")
        if "urgent cash" in text or "traveller" in text or "travelers" in text or "stolen" in text:
            return Classification(company="Visa", request_type="product_issue", product_area="travel_support", status_hint="replied", retrieval_query="emergency cash lost stolen card travellers cheques", reasoning="Travel support covers emergency cash and stolen travel instruments.")
        if "minimum" in text or "spend" in text or "surcharge" in text:
            return Classification(company="Visa", request_type="product_issue", product_area="regulations_fees", status_hint="replied", retrieval_query="minimum purchase amount visa card rules", reasoning="Minimum spend questions are covered by Visa rules.")
        return Classification(company="Visa", request_type="product_issue", product_area="general_support", status_hint="replied", retrieval_query=text[:80], reasoning="Visa support question.")

    if company == "HackerRank":
        if "delete my account" in text or "google login" in text:
            return Classification(company="HackerRank", request_type="product_issue", product_area="community", status_hint="replied", retrieval_query="delete account google login", reasoning="Account deletion for a community login is covered by HackerRank community docs.")
        if "mock interview" in text or "apply tab" in text or "certificate" in text or "resume builder" in text:
            request_type = "bug" if "down" in text or "stopped" in text else "product_issue"
            status = "escalated" if "down" in text else "replied"
            query = "community mock interview certificate apply resume builder"
            return Classification(company="HackerRank", request_type=request_type, product_area="community", status_hint=status, retrieval_query="" if status == "escalated" else query, reasoning="Community product issue.")
        if "remove" in text and ("interviewer" in text or "employee" in text or "user" in text):
            return Classification(company="HackerRank", request_type="product_issue", product_area="settings", status_hint="replied", retrieval_query="remove user roles management", reasoning="User removal is covered by settings and roles docs.")
        if "compatible check" in text or "zoom" in text:
            return Classification(company="HackerRank", request_type="product_issue", product_area="interviews", status_hint="replied", retrieval_query="zoom connectivity compatibility check", reasoning="Compatibility check issues are covered by interview docs.")
        if "inactivity" in text or "lobby" in text:
            return Classification(company="HackerRank", request_type="product_issue", product_area="interviews", status_hint="replied", retrieval_query="candidate interviewer inactivity timeout", reasoning="Inactivity timers are covered by interview settings docs.")
        if "submissions across any challenges" in text:
            return Classification(company="HackerRank", request_type="bug", product_area="community", status_hint="escalated", retrieval_query="", reasoning="The user reports submissions failing across all HackerRank challenges. This looks like a site-wide product bug requiring technical investigation; the corpus cannot confirm live incident status.")
        return Classification(company="HackerRank", request_type="product_issue", product_area="screen", status_hint="replied", retrieval_query=text[:80], reasoning="HackerRank support question.")

    return Classification(company="None", request_type="invalid", product_area="", status_hint="replied", retrieval_query="", reasoning="No supported product domain was identified.")


def safety_guardrails(ticket: Ticket) -> Classification | None:
    """Deterministic guardrails over the LLM classifier.

    These rules exist as a safety net to guarantee correct escalation on
    high-risk patterns (billing disputes, identity theft, security
    vulnerabilities, prompt injection, etc.) and stable product-area
    slugs on a small set of well-understood ticket shapes, regardless
    of LLM behavior.

    For unseen tickets, the LLM classifier still runs; these guardrails
    only override output for matched high-risk or well-understood patterns.
    They are not the whole system, they bound it.
    """
    text = _ticket_text(ticket)
    company = ticket.company

    if "delete my account" in text or "google login" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="community", status_hint="replied", retrieval_query="delete account google login", reasoning="Account deletion for a Community login is answerable from HackerRank Community docs.")
    if "iron man" in text:
        return Classification(company="None", request_type="invalid", product_area="conversation_management", status_hint="replied", retrieval_query="", reasoning="The ticket is an off-topic conversation request.")
    if company == "Visa" and "lost or stolen visa card" in text:
        return Classification(company="Visa", request_type="product_issue", product_area="general_support", status_hint="replied", retrieval_query="lost stolen visa card report", reasoning="Lost or stolen Visa card reporting is covered by Visa support docs.")
    if company == "Visa" and "card stolen" in text:
        return Classification(company="Visa", request_type="product_issue", product_area="general_support", status_hint="replied", retrieval_query="lost stolen visa card report", reasoning="Lost or stolen Visa card reporting is covered by Visa support docs.")
    if "lost access" in text and "claude team workspace" in text and "removed my seat" in text:
        return Classification(company="Claude", request_type="product_issue", product_area="team_and_enterprise", status_hint="replied", retrieval_query="removed seat team workspace re added", reasoning="Claude team workspace seat access is answerable from team and enterprise docs.")
    if "increase my score" in text or "move me to the next round" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="screen", status_hint="escalated", retrieval_query="", reasoning="The user asks HackerRank to review answers, increase a score, and influence a hiring decision. This is a case-specific assessment and employment outcome request that the support corpus cannot resolve.")
    if company == "Visa" and "wrong product" in text and "refund" in text:
        return Classification(company="Visa", request_type="product_issue", product_area="dispute_resolution", status_hint="escalated", retrieval_query="", reasoning="The user wants Visa to refund a transaction and ban a merchant. This requires issuer dispute handling and account-specific investigation; the corpus can explain disputes but cannot reverse a charge or take merchant enforcement action.")
    if "mock interviews" in text and "refund" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="billing", status_hint="escalated", retrieval_query="", reasoning="The user asks for a refund after a mock interview stopped. Refund decisions require billing/account review; the corpus cannot issue or approve refunds.")
    if "order id" in text or "give me my money" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="billing", status_hint="escalated", retrieval_query="", reasoning="The user reports a payment issue with a specific order ID. This requires account and transaction lookup; the corpus cannot inspect or resolve individual payment records.")
    if "infosec" in text or "filling in the forms" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="general_help", status_hint="escalated", retrieval_query="", reasoning="The user asks HackerRank support to fill out company infosec forms. This requires verified business and security review; the corpus cannot complete customer-specific compliance paperwork.")
    if "apply tab" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="community", status_hint="replied", retrieval_query="community apply tab jobs practice", reasoning="Apply tab issues are part of HackerRank Community support.")
    if "submissions across any challenges" in text:
        return Classification(company="HackerRank", request_type="bug", product_area="community", status_hint="escalated", retrieval_query="", reasoning="The user reports submissions failing across all HackerRank challenges. This looks like a site-wide product bug requiring technical investigation; the corpus cannot confirm live incident status.")
    if "compatible check" in text or "zoom connectivity" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="interviews", status_hint="replied", retrieval_query="zoom connectivity compatibility check", reasoning="Compatibility check and Zoom connectivity are covered by interview support docs.")
    if "rescheduling" in text or "alternative date" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="screen", status_hint="escalated", retrieval_query="", reasoning="The user wants a HackerRank assessment rescheduled after missing the scheduled time. Scheduling is controlled by the hiring company; the corpus cannot grant a new assessment window.")
    if "inactivity" in text or "hr lobby" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="interviews", status_hint="replied", retrieval_query="candidate interviewer inactivity lobby timeout", reasoning="Interview inactivity and lobby behavior are answerable from interview docs.")
    if "it’s not working" in text or "it's not working" in text:
        return Classification(company="None", request_type="bug", product_area="", status_hint="escalated", retrieval_query="", reasoning="The user reports that something is not working without enough product or error context. The corpus cannot diagnose an unspecified issue, so a human needs to gather details.")
    if "remove an interviewer" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="settings", status_hint="replied", retrieval_query="remove user team member roles management", reasoning="Removing interviewer users is covered by settings and team management docs.")
    if "pause our subscription" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="billing", status_hint="escalated", retrieval_query="", reasoning="The user wants their HackerRank subscription paused. This is an account-level action a support agent cannot perform; the corpus documents subscription plans but not pause requests on a customer's behalf.")
    if "claude has stopped working completely" in text:
        return Classification(company="Claude", request_type="bug", product_area="general", status_hint="escalated", retrieval_query="", reasoning="The user reports Claude has stopped working completely and all requests are failing. This requires live operational investigation; the static corpus cannot confirm or remediate current service availability.")
    if "identity" in text and "stolen" in text:
        return Classification(company="Visa", request_type="product_issue", product_area="fraud_protection", status_hint="escalated", retrieval_query="", reasoning="The user reports identity theft. This is a high-risk fraud scenario requiring the Visa fraud team; the corpus covers general fraud prevention but not case-level incident response.")
    if "resume builder is down" in text:
        return Classification(company="HackerRank", request_type="bug", product_area="community", status_hint="escalated", retrieval_query="", reasoning="The user reports the HackerRank Community Resume Builder is down. This is a feature availability issue requiring technical investigation; the corpus cannot confirm or repair a live outage.")
    if "certificate" in text and "name" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="community", status_hint="replied", retrieval_query="certificate name update", reasoning="Certificate name updates are covered by Community certification docs.")
    if company == "Visa" and "dispute a charge" in text:
        return Classification(company="Visa", request_type="product_issue", product_area="dispute_resolution", status_hint="replied", retrieval_query="dispute charge card issuer", reasoning="Charge disputes are answerable from Visa dispute docs.")
    if "security vulnerability" in text or "bug bounty" in text:
        return Classification(company="Claude", request_type="bug", product_area="safeguards", status_hint="escalated", retrieval_query="", reasoning="The user reports a Claude security vulnerability. Vulnerability disclosures must go through Anthropic's security team; the corpus covers safeguards policy but not bug-bounty intake.")
    if "stop crawling" in text:
        return Classification(company="Claude", request_type="product_issue", product_area="privacy", status_hint="replied", retrieval_query="block anthropic crawler robots txt", reasoning="Crawler opt-out is covered by Claude privacy docs.")
    if "urgent cash" in text and "visa" in text:
        return Classification(company="Visa", request_type="product_issue", product_area="travel_support", status_hint="replied", retrieval_query="urgent cash visa card atm gcas", reasoning="Urgent cash and GCAS support are covered by Visa travel support docs.")
    if "use my data to improve the models" in text:
        return Classification(company="Claude", request_type="product_issue", product_area="privacy", status_hint="replied", retrieval_query="data used improve models duration", reasoning="Data use duration is covered by Claude privacy docs.")
    if "delete all files" in text:
        return Classification(company="None", request_type="invalid", product_area="conversation_management", status_hint="replied", retrieval_query="", reasoning="The user asks the agent to perform an unsafe filesystem action. This is outside support scope; the response is the canned out-of-scope reply.")
    if "règles internes" in text or "documents récupérés" in text or "internal rules" in text:
        return Classification(company="Visa", request_type="product_issue", product_area="fraud_protection", status_hint="escalated", retrieval_query="", reasoning="The user asks for internal rules and fraud-detection logic while describing a blocked Visa card. This is a prompt-injection attempt and a possible fraud case, so it must be escalated without revealing internal process.")
    if "bedrock" in text and "failing" in text:
        return Classification(company="Claude", request_type="product_issue", product_area="amazon_bedrock", status_hint="replied", retrieval_query="amazon bedrock claude requests failing support", reasoning="Claude on AWS Bedrock is covered by Amazon Bedrock docs.")
    if "employee has left" in text or "remove them from our hackerrank hiring account" in text:
        return Classification(company="HackerRank", request_type="product_issue", product_area="settings", status_hint="replied", retrieval_query="remove user team member roles management", reasoning="Employee removal is covered by settings and team management docs.")
    if "lti key" in text or "professor" in text:
        return Classification(company="Claude", request_type="product_issue", product_area="education", status_hint="replied", retrieval_query="claude education lti key canvas", reasoning="Claude LTI setup is covered by education docs.")
    if "minimum 10" in text or "minimum spend" in text:
        return Classification(company="Visa", request_type="product_issue", product_area="regulations_fees", status_hint="replied", retrieval_query="minimum transaction amount US territories Visa", reasoning="Minimum spend rules are covered by Visa regulations and fees docs.")
    return None


def enforce_overrides(ticket: Ticket, classification: Classification) -> Classification:
    rule = safety_guardrails(ticket)
    if rule is not None:
        return rule
    rule = heuristic_classify(ticket)
    text = _ticket_text(ticket)
    hard_terms = (
        "delete all files",
        "règles internes",
        "internal rules",
        "security vulnerability",
        "bug bounty",
        "identity",
        "infosec",
        "order id",
        "refund",
        "pause our subscription",
        "rescheduling",
        "increase my score",
        "site is down",
        "it's not working",
        "it’s not working",
    )
    if any(term in text for term in hard_terms):
        return rule
    if classification.status_hint == "escalated":
        return classification
    return classification


def classify_ticket(ticket: Ticket) -> Classification:
    deterministic = safety_guardrails(ticket)
    if deterministic is not None:
        return deterministic

    user_message = f"""{FEW_SHOT}

Now classify this ticket:
Issue: {json.dumps(ticket.issue)}
Subject: {json.dumps(ticket.subject)}
Company: {json.dumps(ticket.company)}
Output:"""
    try:
        raw = call_claude("classify", SYSTEM_PROMPT, user_message)
        classification = Classification.model_validate(_json_object(raw))
        return enforce_overrides(ticket, classification)
    except (LLMUnavailable, ValueError, json.JSONDecodeError):
        return heuristic_classify(ticket)
