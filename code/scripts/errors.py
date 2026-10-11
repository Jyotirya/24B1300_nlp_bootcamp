"""Validation error analysis for SciFact verification predictions."""

from __future__ import annotations

import re
from typing import Collection

from .data import ClaimRecord, Corpus

ALLOWED_CATEGORIES = (
    "retrieval_miss",
    "negation",
    "numerical",
    "entity_mismatch",
    "needs_multiple_docs",
    "other",
)
NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|without|cannot|fails?\s+to|lack|absence)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"(?:\d+(?:\.\d+)?%?)")


def has_negation_cue(text: str) -> bool:
    """Detect whole-word negation cues without substring matching."""
    return bool(NEGATION_PATTERN.search(text))


def classify_error(
    claim: ClaimRecord,
    gold_pairs: Collection[tuple[int, str]],
    predicted_pairs: Collection[tuple[int, str]],
    retrieved_doc_ids: Collection[int],
    corpus: Corpus,
) -> str:
    """Assign a conservative category to one incorrect validation claim."""
    gold_documents = {doc_id for doc_id, _ in gold_pairs}
    predicted_documents = {doc_id for doc_id, _ in predicted_pairs}
    if gold_documents and not gold_documents.intersection(set(retrieved_doc_ids)):
        return "retrieval_miss"
    if not gold_documents and predicted_pairs:
        return "other"
    if re.search(r"\d|%", claim.claim):
        return "numerical"
    if has_negation_cue(claim.claim):
        return "negation"
    if len(gold_documents) >= 2:
        return "needs_multiple_docs"
    if any(doc_id not in gold_documents for doc_id in predicted_documents):
        return "entity_mismatch"
    return "other"
