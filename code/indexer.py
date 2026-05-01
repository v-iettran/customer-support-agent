from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

from rank_bm25 import BM25Okapi

from models import CorpusStats, RetrievedChunk


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_'][a-z0-9]+)?", re.IGNORECASE)
TITLE_RE = re.compile(r'^title:\s*"?([^"\n]+)"?', re.MULTILINE)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def strip_frontmatter(content: str) -> tuple[str, str]:
    if not content.startswith("---"):
        return "", content.strip()
    end = content.find("\n---", 3)
    if end == -1:
        return "", content.strip()
    frontmatter = content[3:end]
    body = content[end + 4 :].strip()
    match = TITLE_RE.search(frontmatter)
    title = match.group(1).strip() if match else ""
    return title, body


def category_from_path(domain: str, parts: tuple[str, ...]) -> str:
    if domain == "visa":
        if "travel-support" in parts or any("travelers-cheques" in part for part in parts):
            return "travel_support"
        if any(part in {"visa-rules.md", "checkout-fees-contact-form.md", "regulations-fees.md"} for part in parts):
            return "regulations_fees"
        if "dispute-resolution.md" in parts:
            return "dispute_resolution"
        if "fraud-protection.md" in parts:
            return "fraud_protection"
        if "data-security.md" in parts:
            return "data_security"
        return "general_support"
    if len(parts) < 2:
        return ""
    category = parts[1]
    mapping = {
        "hackerrank_community": "community",
        "general-help": "general_help",
        "privacy-and-legal": "privacy",
        "team-and-enterprise-plans": "team_and_enterprise",
        "pro-and-max-plans": "pro_and_max",
        "claude-code": "claude_code",
        "claude-api-and-console": "claude_api",
        "claude-desktop": "claude_desktop",
        "claude-for-education": "education",
        "amazon-bedrock": "amazon_bedrock",
        "claude-mobile-apps": "mobile",
        "claude-in-chrome": "chrome",
        "claude": "general",
    }
    return mapping.get(category, category.replace("-", "_"))


def split_sections(body: str) -> list[str]:
    body = body.replace("\r\n", "\n")
    sections = re.split(r"\n(?=#{1,3}\s)|\n{2,}", body)
    return [section.strip() for section in sections if section.strip()]


def chunk_text(body: str, target_words: int = 300) -> list[str]:
    sections = split_sections(body)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_words = 0

    for section in sections:
        words = section.split()
        if len(words) > target_words:
            flush()
            for start in range(0, len(words), target_words):
                chunks.append(" ".join(words[start : start + target_words]).strip())
            continue
        if current_words + len(words) > target_words:
            flush()
        current.append(section)
        current_words += len(words)
    flush()
    return chunks


def load_corpus(data_dir: Path) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for root, _, files in os.walk(data_dir):
        for filename in sorted(files):
            if not filename.endswith(".md"):
                continue
            file_path = Path(root) / filename
            rel_parts = file_path.relative_to(data_dir).parts
            if not rel_parts:
                continue
            domain = rel_parts[0]
            content = file_path.read_text(encoding="utf-8")
            title, body = strip_frontmatter(content)
            category = category_from_path(domain, rel_parts)
            for chunk in chunk_text(body):
                chunks.append(
                    RetrievedChunk(
                        text=chunk,
                        title=title or file_path.stem.replace("-", " "),
                        domain=domain,
                        category=category,
                        file_path=str(file_path),
                    )
                )
    return chunks


class CorpusIndex:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index with no chunks")
        self.chunks = chunks
        tokenized = [tokenize(f"{chunk.title} {chunk.category} {chunk.text}") for chunk in chunks]
        self.index = BM25Okapi(tokenized)

    @property
    def stats(self) -> CorpusStats:
        domains = Counter(chunk.domain for chunk in self.chunks)
        files = len({chunk.file_path for chunk in self.chunks})
        return CorpusStats(files=files, chunks=len(self.chunks), domains=dict(domains))

    def search(self, query: str, domain_filter: str | None = None, top_k: int = 5) -> list[RetrievedChunk]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = list(self.index.get_scores(tokens))
        target = (domain_filter or "").strip().lower()
        domain_map = {"hackerrank": "hackerrank", "claude": "claude", "visa": "visa"}
        if target in domain_map:
            for index, chunk in enumerate(self.chunks):
                if chunk.domain != domain_map[target]:
                    scores[index] = -1.0
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: list[RetrievedChunk] = []
        for index, score in ranked:
            if score <= 0:
                continue
            chunk = self.chunks[index]
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    title=chunk.title,
                    domain=chunk.domain,
                    category=chunk.category,
                    file_path=chunk.file_path,
                    score=float(score),
                )
            )
            if len(results) >= top_k:
                break
        return results


def build_corpus_index(data_dir: Path) -> CorpusIndex:
    return CorpusIndex(load_corpus(data_dir))
