from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator


Status = Literal["replied", "escalated"]
RequestType = Literal["product_issue", "feature_request", "bug", "invalid"]
Company = Literal["HackerRank", "Claude", "Visa", "None"]

ALLOWED_PRODUCT_AREAS = {
    "",
    "amazon_bedrock",
    "billing",
    "chakra",
    "claude_api",
    "claude_code",
    "claude_desktop",
    "community",
    "connectors",
    "conversation_management",
    "data_security",
    "dispute_resolution",
    "education",
    "engage",
    "fraud_protection",
    "general",
    "general_help",
    "general_support",
    "integrations",
    "interviews",
    "library",
    "privacy",
    "pro_and_max",
    "regulations_fees",
    "safeguards",
    "screen",
    "settings",
    "skillup",
    "team_and_enterprise",
    "travel_support",
}

PRODUCT_AREA_ALIASES = {
    "account_management": "settings",
    "api": "claude_api",
    "aws_bedrock": "amazon_bedrock",
    "cash_advance": "travel_support",
    "certificate": "community",
    "certificates": "community",
    "challenges": "community",
    "data_privacy": "privacy",
    "disputes": "dispute_resolution",
    "interview": "interviews",
    "merchant_policies": "regulations_fees",
    "practice": "community",
    "privacy_data": "privacy",
    "resume_builder": "community",
    "test_taking": "interviews",
    "workspace_management": "team_and_enterprise",
}


def normalize_product_area_slug(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    slug = raw.lower().replace("-", "_").replace(" ", "_")
    slug = PRODUCT_AREA_ALIASES.get(slug, slug)
    return slug if slug in ALLOWED_PRODUCT_AREAS else ""


class Ticket(BaseModel):
    issue: str
    subject: str = ""
    company: Company = "None"
    row_number: int | None = None

    @field_validator("issue", "subject", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("company", mode="before")
    @classmethod
    def normalize_company(cls, value: object) -> str:
        raw = "" if value is None else str(value).strip()
        if raw.lower() == "none" or raw == "":
            return "None"
        for company in ("HackerRank", "Claude", "Visa"):
            if raw.lower() == company.lower():
                return company
        return "None"


class Classification(BaseModel):
    company: Company = "None"
    request_type: RequestType
    product_area: str = ""
    status_hint: Status
    retrieval_query: str = ""
    reasoning: str = ""

    @field_validator("product_area", "retrieval_query", "reasoning", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("status_hint", mode="before")
    @classmethod
    def normalize_status(cls, value: object) -> str:
        return str(value).strip().lower()

    @field_validator("request_type", mode="before")
    @classmethod
    def normalize_request_type(cls, value: object) -> str:
        return str(value).strip().lower()

    @field_validator("product_area")
    @classmethod
    def slugify_product_area(cls, value: str) -> str:
        return normalize_product_area_slug(value)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    title: str
    domain: str
    category: str
    file_path: str
    score: float = 0.0


class SafetyResult(BaseModel):
    final_status: Status
    revised_response: str
    revision_notes: str = "no changes"

    @field_validator("final_status", mode="before")
    @classmethod
    def normalize_status(cls, value: object) -> str:
        return str(value).strip().lower()


class OutputRow(BaseModel):
    issue: str
    subject: str
    company: Company
    response: str
    product_area: str = ""
    status: Status
    request_type: RequestType
    justification: str

    @field_validator("status", "request_type", mode="before")
    @classmethod
    def lowercase_values(cls, value: object) -> str:
        return str(value).strip().lower()

    @field_validator("product_area", mode="before")
    @classmethod
    def normalize_product_area(cls, value: object) -> str:
        return normalize_product_area_slug(value)


class CorpusStats(BaseModel):
    files: int = 0
    chunks: int = 0
    domains: dict[str, int] = Field(default_factory=dict)
