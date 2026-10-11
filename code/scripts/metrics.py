"""Retrieval metrics and TREC run-file utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

from .data import Claims

Ranking = Sequence[tuple[int, float]]
GoldDocuments = Mapping[int, Collection[int]]


def _gold_set(gold_doc_ids: Iterable[int]) -> set[int]:
    return set(gold_doc_ids)


def ndcg_at_k(
    ranked_doc_ids: Sequence[int],
    gold_doc_ids: Collection[int],
    k: int = 10,
) -> float:
    """Return binary-relevance nDCG at ``k`` for one claim."""
    if k <= 0:
        raise ValueError("k must be positive")
    gold = _gold_set(gold_doc_ids)
    if not gold:
        return 0.0
    retrieved_gain = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc_id in enumerate(ranked_doc_ids[:k])
        if doc_id in gold
    )
    ideal_gain = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(gold))))
    return retrieved_gain / ideal_gain if ideal_gain else 0.0


def recall_at_k(
    ranked_doc_ids: Sequence[int],
    gold_doc_ids: Collection[int],
    k: int = 100,
) -> float:
    """Return the fraction of gold documents retrieved in the first ``k``."""
    if k <= 0:
        raise ValueError("k must be positive")
    gold = _gold_set(gold_doc_ids)
    return len(gold.intersection(ranked_doc_ids[:k])) / len(gold) if gold else 0.0


def reciprocal_rank(
    ranked_doc_ids: Sequence[int],
    gold_doc_ids: Collection[int],
) -> float:
    """Return the reciprocal rank of the first retrieved gold document."""
    gold = _gold_set(gold_doc_ids)
    if not gold:
        return 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def gold_documents(claims: Claims) -> dict[int, set[int]]:
    """Extract unique gold document IDs from loaded claim records."""
    return {
        record.claim_id: {item.doc_id for item in record.evidence}
        for record in claims.records
    }


def evaluate_retrieval(
    rankings: Mapping[int, Sequence[int]],
    gold: GoldDocuments,
    *,
    ndcg_k: int = 10,
    recall_k: int = 100,
) -> dict[str, object]:
    """Evaluate rankings and return per-claim values plus non-empty-gold means.

    Claims with no gold documents are included in ``per_claim`` as zeroes but
    excluded from all aggregate means.
    """
    per_claim: dict[int, dict[str, float]] = {}
    aggregate_ids: list[int] = []
    for claim_id, gold_doc_ids in gold.items():
        ranked_doc_ids = rankings.get(claim_id, [])
        per_claim[claim_id] = {
            "ndcg@10": ndcg_at_k(ranked_doc_ids, gold_doc_ids, ndcg_k),
            "recall@100": recall_at_k(ranked_doc_ids, gold_doc_ids, recall_k),
            "mrr": reciprocal_rank(ranked_doc_ids, gold_doc_ids),
        }
        if gold_doc_ids:
            aggregate_ids.append(claim_id)
    means = {
        metric: (
            sum(per_claim[claim_id][metric] for claim_id in aggregate_ids)
            / len(aggregate_ids)
            if aggregate_ids
            else 0.0
        )
        for metric in ("ndcg@10", "recall@100", "mrr")
    }
    return {"per_claim": per_claim, "mean": means}


def _validate_claim_id(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _strict_scores(scores: Sequence[float]) -> list[float]:
    """Keep descending scores, or use descending fallback scores in same order."""
    numeric = [float(score) for score in scores]
    if all(math.isfinite(score) for score in numeric) and all(
        left > right for left, right in zip(numeric, numeric[1:])
    ):
        return numeric
    return [float(len(numeric) - rank) for rank in range(len(numeric))]


def write_trec(
    path: str | Path,
    rankings: Mapping[int, Ranking],
    run_name: str,
    *,
    expected_claim_ids: Collection[int] | None = None,
    top_k: int = 100,
) -> None:
    """Write ranked ``(doc_id, score)`` pairs as six-field TREC rows.

    Each claim must have exactly ``top_k`` unique integer document IDs. The
    default ``top_k=100`` matches the submission format. Input order is never
    changed; scores are replaced only when needed to make them strictly
    decrease by rank.
    """
    if top_k != 100:
        raise ValueError("submission TREC files require top_k=100")
    if not run_name or any(char.isspace() for char in run_name):
        raise ValueError("run_name must be a non-empty token")
    if expected_claim_ids is not None and set(rankings) != set(expected_claim_ids):
        raise ValueError("rankings must contain exactly the expected claim IDs")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for claim_id, ranking in rankings.items():
            _validate_claim_id(claim_id, "claim_id")
            if len(ranking) != top_k:
                raise ValueError(f"claim {claim_id} must have exactly {top_k} documents")
            doc_ids = [doc_id for doc_id, _ in ranking]
            if any(isinstance(doc_id, bool) or not isinstance(doc_id, int) for doc_id in doc_ids):
                raise ValueError(f"claim {claim_id} contains a non-integer document ID")
            if len(set(doc_ids)) != top_k:
                raise ValueError(f"claim {claim_id} contains duplicate document IDs")
            scores = _strict_scores([score for _, score in ranking])
            for rank, ((doc_id, _), score) in enumerate(zip(ranking, scores), 1):
                handle.write(f"{claim_id} Q0 {doc_id} {rank} {score:.12g} {run_name}\n")


def read_trec(
    path: str | Path,
    *,
    expected_claim_ids: Collection[int] | None = None,
    expected_doc_ids: Collection[int] | None = None,
    top_k: int = 100,
) -> dict[int, list[tuple[int, float]]]:
    """Read and validate a six-field TREC file, preserving rank order."""
    if top_k != 100:
        raise ValueError("submission TREC files require top_k=100")
    trec_path = Path(path)
    if not trec_path.is_file():
        raise FileNotFoundError(f"TREC file not found: {trec_path}")
    rows: dict[int, dict[int, tuple[int, float]]] = {}
    run_name: str | None = None
    with trec_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"{trec_path}:{line_number} must have 6 fields")
            try:
                claim_id, q0, doc_id, rank = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
                score = float(fields[4])
            except ValueError as exc:
                raise ValueError(f"{trec_path}:{line_number} has invalid numeric fields") from exc
            if q0 != "Q0" or not math.isfinite(score):
                raise ValueError(f"{trec_path}:{line_number} has invalid Q0 or score")
            if expected_claim_ids is not None and claim_id not in expected_claim_ids:
                raise ValueError(f"{trec_path}:{line_number} has unknown claim_id {claim_id}")
            if expected_doc_ids is not None and doc_id not in expected_doc_ids:
                raise ValueError(f"{trec_path}:{line_number} has unknown doc_id {doc_id}")
            if rank < 1 or rank > top_k:
                raise ValueError(f"{trec_path}:{line_number} rank must be 1..{top_k}")
            if run_name is None:
                run_name = fields[5]
            elif fields[5] != run_name:
                raise ValueError(f"{trec_path}:{line_number} has inconsistent run names")
            claim_rows = rows.setdefault(claim_id, {})
            if rank in claim_rows:
                raise ValueError(f"{trec_path}:{line_number} repeats rank {rank} for claim {claim_id}")
            if any(existing_doc_id == doc_id for existing_doc_id, _ in claim_rows.values()):
                raise ValueError(f"{trec_path}:{line_number} repeats doc_id {doc_id} for claim {claim_id}")
            claim_rows[rank] = (doc_id, score)
    if expected_claim_ids is not None and set(rows) != set(expected_claim_ids):
        raise ValueError("TREC file is missing claims")
    output: dict[int, list[tuple[int, float]]] = {}
    for claim_id, claim_rows in rows.items():
        if set(claim_rows) != set(range(1, top_k + 1)):
            raise ValueError(f"claim {claim_id} must have ranks 1..{top_k}")
        output[claim_id] = [claim_rows[rank] for rank in range(1, top_k + 1)]
    return output
