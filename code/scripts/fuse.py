"""Hybrid fusion utilities for SciFact retrieval runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

Ranking = Sequence[tuple[int, float]]
Run = Mapping[int, Ranking]


def _minmax_scores(ranking: Ranking, depth: int) -> dict[int, float]:
    values = {doc_id: float(score) for doc_id, score in ranking[:depth]}
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {doc_id: 0.0 for doc_id in values}
    return {doc_id: (score - low) / (high - low) for doc_id, score in values.items()}


def rrf_fuse(
    runs: Sequence[Run],
    *,
    weights: Sequence[float] | None = None,
    constant: int = 60,
    depth: int = 100,
) -> dict[int, list[tuple[int, float]]]:
    """Fuse runs with weighted reciprocal rank fusion."""
    if not runs:
        raise ValueError("at least one run is required")
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    if weights is None:
        weights = [1.0] * len(runs)
    if len(weights) != len(runs) or any(weight < 0 for weight in weights):
        raise ValueError("weights must match runs and be non-negative")
    claim_ids = sorted(set().union(*(run.keys() for run in runs)))
    output: dict[int, list[tuple[int, float]]] = {}
    for claim_id in claim_ids:
        fused: dict[int, float] = {}
        for run, weight in zip(runs, weights):
            for rank, (doc_id, _) in enumerate(run.get(claim_id, [])[:depth], 1):
                fused[doc_id] = fused.get(doc_id, 0.0) + weight / (constant + rank)
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        output[claim_id] = ordered[: max(100, depth)]
    return output


def compare_rrf_constants(
    runs: Sequence[Run],
    validation_gold: Mapping[int, Collection[int]],
    constants: Sequence[int],
    *,
    depth: int = 200,
) -> tuple[int, dict[int, float]]:
    """Select an RRF constant by validation nDCG@10."""
    from .metrics import evaluate_retrieval

    scores: dict[int, float] = {}
    for constant in constants:
        fused = rrf_fuse(runs, constant=constant, depth=depth)
        scores[constant] = float(evaluate_retrieval(
            {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in fused.items()},
            validation_gold,
        )["mean"]["ndcg@10"])
    best = max(scores, key=lambda value: (scores[value], -value))
    return best, scores


def search_hybrid(
    bm25_run: Run,
    general_run: Run,
    gold: Mapping[int, Collection[int]],
    *,
    rrf_constants: Sequence[int],
    dense_weights: Sequence[float],
    interpolation_alphas: Sequence[float],
    depth: int = 200,
) -> tuple[float, dict[str, Any]]:
    """Grid-search weighted RRF and score interpolation by validation nDCG@10."""
    from .metrics import evaluate_retrieval

    def ndcg(run: Run) -> float:
        return float(evaluate_retrieval(
            {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in run.items()},
            gold,
        )["mean"]["ndcg@10"])

    if not rrf_constants or not dense_weights or not interpolation_alphas:
        raise ValueError("hybrid search candidate lists must not be empty")

    best: tuple[float, dict[str, Any]] | None = None
    for constant in rrf_constants:
        for dense_weight in dense_weights:
            spec = {
                "method": "rrf",
                "rrf_constant": constant,
                "weights": [1.0, dense_weight],
            }
            score = ndcg(apply_hybrid(spec, bm25_run, general_run, depth=depth))
            if best is None or score > best[0]:
                best = (score, spec)
    for alpha in interpolation_alphas:
        spec = {"method": "interpolation", "alpha": alpha}
        score = ndcg(apply_hybrid(spec, bm25_run, general_run, depth=depth))
        if score > best[0]:
            best = (score, spec)
    if best is None:
        raise RuntimeError("hybrid search produced no candidates")
    return best


def apply_hybrid(
    spec: Mapping[str, Any],
    bm25_run: Run,
    general_run: Run,
    *,
    depth: int = 200,
) -> dict[int, list[tuple[int, float]]]:
    """Apply a spec returned by ``search_hybrid`` to a (BM25, dense) run pair."""
    if spec["method"] == "rrf":
        return rrf_fuse(
            [bm25_run, general_run],
            weights=tuple(spec["weights"]),
            constant=int(spec["rrf_constant"]),
            depth=depth,
        )
    if spec["method"] == "interpolation":
        return interpolate_fuse(
            general_run, bm25_run, alpha=float(spec["alpha"]), depth=depth
        )
    raise ValueError(f"unsupported hybrid method: {spec['method']}")


def interpolate_fuse(
    dense_run: Run,
    bm25_run: Run,
    *,
    alpha: float = 0.5,
    depth: int = 100,
) -> dict[int, list[tuple[int, float]]]:
    """Fuse min-max-normalized dense and BM25 scores per claim."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    claim_ids = sorted(set(dense_run) | set(bm25_run))
    output: dict[int, list[tuple[int, float]]] = {}
    for claim_id in claim_ids:
        dense_scores = _minmax_scores(dense_run.get(claim_id, []), depth)
        bm25_scores = _minmax_scores(bm25_run.get(claim_id, []), depth)
        documents = sorted(set(dense_scores) | set(bm25_scores))
        scores = {
            doc_id: alpha * dense_scores.get(doc_id, 0.0)
            + (1 - alpha) * bm25_scores.get(doc_id, 0.0)
            for doc_id in documents
        }
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        output[claim_id] = ordered[: max(100, depth)]
    return output


def fuse_runs(
    bm25_run: Run,
    general_dense_run: Run,
    *,
    method: str = "rrf",
    scientific_dense_run: Run | None = None,
    depth: int = 100,
    rrf_constant: int = 60,
    weights: Sequence[float] = (1.0, 1.0),
    alpha: float = 0.5,
) -> dict[int, list[tuple[int, float]]]:
    """Fuse BM25 and general dense runs, optionally including scientific dense."""
    runs = [bm25_run, general_dense_run]
    if scientific_dense_run is not None:
        runs.append(scientific_dense_run)
        if len(weights) == 2:
            weights = (*weights, 1.0)
    if method == "rrf":
        return rrf_fuse(runs, weights=weights, constant=rrf_constant, depth=depth)
    if method == "interpolation":
        if scientific_dense_run is not None:
            raise ValueError("interpolation accepts only BM25 and general dense runs")
        return interpolate_fuse(general_dense_run, bm25_run, alpha=alpha, depth=depth)
    raise ValueError(f"unsupported fusion method: {method}")


def build_extra_hybrid_ft(
    config: Mapping[str, Any],
    validation_runs: Mapping[str, Run],
    evaluation_runs: Mapping[str, Run],
    validation_claims: Any,
    evaluation_claims: Any,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[tuple[int, float]]], dict[str, Any]]:
    """Build the optional Task 4-bi-encoder hybrid and persist result-side runs."""
    from .metrics import evaluate_retrieval, gold_documents, write_trec

    required = ("bm25", "dense_general", "dense_scientific", "t4_random", "t4_hard")
    missing = [
        name for name in required
        if name not in validation_runs or name not in evaluation_runs
    ]
    if missing:
        raise ValueError(f"extra fine-tuned hybrid requires runs: {', '.join(missing)}")

    gold = gold_documents(validation_claims)
    validation_scores = {
        name: float(evaluate_retrieval(
            {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in run.items()},
            gold,
        )["mean"]["ndcg@10"])
        for name, run in (
            ("t4_random", validation_runs["t4_random"]),
            ("t4_hard", validation_runs["t4_hard"]),
        )
    }
    ft_name = max(validation_scores, key=lambda name: (validation_scores[name], name))
    ft_bi = validation_runs[ft_name]
    ft_bi_eval = evaluation_runs[ft_name]

    source_runs = (
        validation_runs["bm25"],
        ft_bi,
        validation_runs["dense_scientific"],
    )
    source_eval_runs = (
        evaluation_runs["bm25"],
        ft_bi_eval,
        evaluation_runs["dense_scientific"],
    )
    general_source = (
        validation_runs["bm25"],
        validation_runs["dense_general"],
        validation_runs["dense_scientific"],
        ft_bi,
    )
    general_eval_source = (
        evaluation_runs["bm25"],
        evaluation_runs["dense_general"],
        evaluation_runs["dense_scientific"],
        ft_bi_eval,
    )
    fusion_config = config["fusion"]
    constants = tuple(int(value) for value in fusion_config["rrf_constants"])
    depth = int(fusion_config["candidate_depth"])
    candidates: list[tuple[str, int, float, dict[int, list[tuple[int, float]]], dict[int, list[tuple[int, float]]]]] = []
    for variant, runs, eval_runs in (
        ("bm25_ft_bi_scientific", source_runs, source_eval_runs),
        ("bm25_general_scientific_ft_bi", general_source, general_eval_source),
    ):
        for constant in constants:
            validation = rrf_fuse(runs, weights=(1.0,) * len(runs), constant=constant, depth=depth)
            evaluation = rrf_fuse(eval_runs, weights=(1.0,) * len(eval_runs), constant=constant, depth=depth)
            score = float(evaluate_retrieval(
                {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in validation.items()},
                gold,
            )["mean"]["ndcg@10"])
            candidates.append((variant, constant, score, validation, evaluation))
    variant, constant, ndcg, validation, evaluation = max(
        candidates,
        key=lambda item: (item[2], -item[1], item[0]),
    )

    results_dir = Path(config["paths"]["results_dir"])
    expected_val_ids = {record.claim_id for record in validation_claims.records}
    expected_eval_ids = {record.claim_id for record in evaluation_claims.records}
    write_trec(
        results_dir / "extra_hybrid_ft.val.trec",
        {
            claim_id: ranking[:100]
            for claim_id, ranking in validation.items()
        },
        "extra_hybrid_ft",
        expected_claim_ids=expected_val_ids,
    )
    write_trec(
        results_dir / "extra_hybrid_ft.eval.trec",
        {
            claim_id: ranking[:100]
            for claim_id, ranking in evaluation.items()
        },
        "extra_hybrid_ft",
        expected_claim_ids=expected_eval_ids,
    )
    metrics = evaluate_retrieval(
        {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in validation.items()},
        gold,
    )
    return validation, evaluation, {
        "selected_ft_bi": ft_name,
        "selected_variant": variant,
        "rrf_constant": constant,
        "ndcg@10": float(metrics["mean"]["ndcg@10"]),
        "recall@100": float(metrics["mean"]["recall@100"]),
    }
