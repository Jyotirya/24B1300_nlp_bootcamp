"""Strict loaders for the supplied SciFact JSONL data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CORPUS_FIELDS = frozenset({"doc_id", "title", "abstract"})
CLAIM_FIELDS = frozenset({"id", "claim", "evidence"})
EVAL_CLAIM_FIELDS = frozenset({"id", "claim"})
EVIDENCE_FIELDS = frozenset({"doc_id", "label", "sentences"})


@dataclass(frozen=True)
class CorpusRecord:
    doc_id: int
    title: str
    abstract: list[str]


@dataclass(frozen=True)
class Evidence:
    doc_id: int
    label: str
    sentences: list[int]


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: int
    claim: str
    evidence: list[Evidence]


@dataclass(frozen=True)
class Corpus:
    records: list[CorpusRecord]
    id_to_index: dict[int, int]


@dataclass(frozen=True)
class Claims:
    records: list[ClaimRecord]
    id_to_index: dict[int, int]


def _read_jsonl(path: str | Path) -> Iterable[tuple[int, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {jsonl_path} at line {line_number}: {exc.msg}"
                ) from exc


def _require_mapping(value: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number} must contain a JSON object")
    return value


def _check_fields(record: dict[str, Any], expected: frozenset[str], path: Path, line: int) -> None:
    unexpected = set(record) - expected
    missing = expected - set(record)
    if unexpected:
        raise ValueError(f"{path}:{line} has unexpected fields: {sorted(unexpected)}")
    if missing:
        raise ValueError(f"{path}:{line} is missing fields: {sorted(missing)}")


def _require_int(value: Any, name: str, path: Path, line: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}:{line} field {name!r} must be an integer")
    return value


def _require_str(value: Any, name: str, path: Path, line: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path}:{line} field {name!r} must be a string")
    return value


def load_corpus(path: str | Path) -> Corpus:
    """Load corpus records and preserve source order and integer document IDs."""
    corpus_path = Path(path)
    records: list[CorpusRecord] = []
    id_to_index: dict[int, int] = {}
    for line, raw in _read_jsonl(corpus_path):
        record = _require_mapping(raw, corpus_path, line)
        _check_fields(record, CORPUS_FIELDS, corpus_path, line)
        doc_id = _require_int(record["doc_id"], "doc_id", corpus_path, line)
        if doc_id in id_to_index:
            raise ValueError(f"{corpus_path}:{line} has duplicate doc_id {doc_id}")
        title = _require_str(record["title"], "title", corpus_path, line)
        abstract = record["abstract"]
        if not isinstance(abstract, list) or not all(isinstance(s, str) for s in abstract):
            raise ValueError(f"{corpus_path}:{line} field 'abstract' must be a list of strings")
        id_to_index[doc_id] = len(records)
        records.append(CorpusRecord(doc_id, title, list(abstract)))
    return Corpus(records, id_to_index)


def load_claims(path: str | Path, *, has_evidence: bool = True) -> Claims:
    """Load claims, including claims whose evidence list is empty."""
    claims_path = Path(path)
    expected_fields = CLAIM_FIELDS if has_evidence else EVAL_CLAIM_FIELDS
    records: list[ClaimRecord] = []
    id_to_index: dict[int, int] = {}
    for line, raw in _read_jsonl(claims_path):
        record = _require_mapping(raw, claims_path, line)
        _check_fields(record, expected_fields, claims_path, line)
        claim_id = _require_int(record["id"], "id", claims_path, line)
        if claim_id in id_to_index:
            raise ValueError(f"{claims_path}:{line} has duplicate claim id {claim_id}")
        claim = _require_str(record["claim"], "claim", claims_path, line)
        raw_evidence = record.get("evidence", [])
        if not isinstance(raw_evidence, list):
            raise ValueError(f"{claims_path}:{line} field 'evidence' must be a list")
        evidence: list[Evidence] = []
        for evidence_number, raw_item in enumerate(raw_evidence):
            item = _require_mapping(raw_item, claims_path, line)
            _check_fields(item, EVIDENCE_FIELDS, claims_path, line)
            doc_id = _require_int(item["doc_id"], "evidence.doc_id", claims_path, line)
            label = _require_str(item["label"], "evidence.label", claims_path, line)
            sentences = item["sentences"]
            if not isinstance(sentences, list) or not all(
                isinstance(sentence, int) and not isinstance(sentence, bool)
                for sentence in sentences
            ):
                raise ValueError(
                    f"{claims_path}:{line} evidence item {evidence_number} "
                    "field 'sentences' must be a list of integers"
                )
            evidence.append(Evidence(doc_id, label, list(sentences)))
        id_to_index[claim_id] = len(records)
        records.append(ClaimRecord(claim_id, claim, evidence))
    return Claims(records, id_to_index)


def load_all_claim_splits(data_dir: str | Path) -> dict[str, Claims]:
    """Load train, validation, and evaluation claims from one data directory."""
    root = Path(data_dir)
    return {
        "train": load_claims(root / "train_claims.jsonl"),
        "val": load_claims(root / "val_claims.jsonl"),
        "eval": load_claims(root / "eval_claims.jsonl", has_evidence=False),
    }
