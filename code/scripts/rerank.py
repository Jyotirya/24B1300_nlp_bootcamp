"""Cross-encoder reranking utilities for SciFact Task 3."""

from __future__ import annotations

import importlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import DEFAULT_CONFIG, load_config, resolve_device, set_global_seed
from .data import Claims, Corpus, load_all_claim_splits, load_corpus
from .dense import MAX_MODEL_PARAMETERS
from .fuse import apply_hybrid, compare_rrf_constants, fuse_runs, search_hybrid
from .metrics import evaluate_retrieval, gold_documents, write_trec

Ranking = Sequence[tuple[int, float]]


@dataclass(frozen=True)
class RerankSettings:
    name: str = "BAAI/bge-reranker-base"
    batch_size: int = 32
    device: str = "auto"
    candidate_k: int = 100
    max_k: int = 30
    max_length: int = 384


def count_cross_encoder_parameters(model: Any) -> int:
    """Count actual parameters exposed by a CrossEncoder or its inner model."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        inner = getattr(model, "model", None)
        parameters = getattr(inner, "parameters", None)
    if not callable(parameters):
        raise TypeError("cross-encoder does not expose parameters()")
    count = sum(int(parameter.numel()) for parameter in parameters())
    if count > MAX_MODEL_PARAMETERS:
        raise ValueError(
            f"cross-encoder has {count:,} parameters, exceeding the "
            f"{MAX_MODEL_PARAMETERS:,}-parameter limit"
        )
    return count


def document_text(corpus: Corpus, doc_id: int) -> str:
    """Return title plus abstract text for a corpus document ID."""
    index = corpus.id_to_index.get(doc_id)
    if index is None:
        raise ValueError(f"unknown document ID: {doc_id}")
    record = corpus.records[index]
    return f"{record.title}\n{' '.join(record.abstract)}"


def rerank_one(
    claim: str,
    ranking: Ranking,
    corpus: Corpus,
    score_pairs: Any,
    *,
    candidate_k: int,
) -> tuple[list[tuple[int, float]], float]:
    """Rerank a prefix and append the untouched remainder in original order."""
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    prefix: list[tuple[int, float]] = []
    seen_prefix: set[int] = set()
    for item in ranking:
        if item[0] not in seen_prefix:
            prefix.append(item)
            seen_prefix.add(item[0])
        if len(prefix) == candidate_k:
            break
    prefix_ids = seen_prefix
    remainder = [(doc_id, score) for doc_id, score in ranking if doc_id not in prefix_ids]
    pairs = [(claim, document_text(corpus, doc_id)) for doc_id, _ in prefix]
    start = time.perf_counter()
    scores = [float(score) for score in score_pairs(pairs)]
    elapsed_ms = (time.perf_counter() - start) * 1000
    if len(scores) != len(prefix):
        raise ValueError("cross-encoder returned the wrong number of scores")
    reranked = sorted(
        zip((doc_id for doc_id, _ in prefix), scores),
        key=lambda item: (-item[1], item[0]),
    )
    seen = {doc_id for doc_id, _ in reranked}
    reranked.extend((doc_id, score) for doc_id, score in remainder if doc_id not in seen)
    return reranked, elapsed_ms


class CrossEncoderReranker:
    """Lazy CrossEncoder wrapper with actual parameter-limit enforcement."""

    def __init__(self, settings: RerankSettings) -> None:
        if settings.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        settings = RerankSettings(**{
            **settings.__dict__,
            "device": resolve_device(settings.device),
        })
        sentence_transformers = importlib.import_module("sentence_transformers")
        self.model = sentence_transformers.CrossEncoder(
            settings.name,
            device=settings.device,
        )
        self.model.max_length = settings.max_length
        self.settings = settings
        self.parameter_count = count_cross_encoder_parameters(self.model)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = self.model.predict(
            list(pairs),
            batch_size=self.settings.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def score_cache(
        self,
        claims: Claims,
        first_stage: Mapping[int, Ranking],
        corpus: Corpus,
        path: str | Path,
    ) -> tuple[dict[str, float], float]:
        """Score each claim's top max_k documents once and cache the scores."""
        cache_path = Path(path)
        cache: dict[str, float] = {}
        if cache_path.is_file():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"invalid cross-encoder cache: {cache_path}")
            cache = {str(key): float(value) for key, value in raw.items()}
        latencies: list[float] = []
        for record in claims.records:
            ranking = list(first_stage.get(record.claim_id, []))[: self.settings.max_k]
            pairs = [(record.claim, document_text(corpus, doc_id)) for doc_id, _ in ranking]
            keys = [f"{record.claim_id}:{doc_id}" for doc_id, _ in ranking]
            if all(key in cache for key in keys):
                continue
            start = time.perf_counter()
            scores = self.score_pairs(pairs)
            cache.update(dict(zip(keys, scores)))
            latencies.append((time.perf_counter() - start) * 1000)
            if len(cache) % 50 == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
        mean = sum(latencies) / len(latencies) if latencies else 0.0
        return cache, mean

    def rerank(
        self,
        claims: Claims,
        first_stage: Mapping[int, Ranking],
        corpus: Corpus,
        *,
        candidate_k: int | None = None,
    ) -> tuple[dict[int, list[tuple[int, float]]], dict[str, float]]:
        k = self.settings.candidate_k if candidate_k is None else candidate_k
        output: dict[int, list[tuple[int, float]]] = {}
        latencies: list[float] = []
        for record in claims.records:
            ranking = first_stage.get(record.claim_id, [])
            reranked, elapsed_ms = rerank_one(
                record.claim,
                ranking,
                corpus,
                self.score_pairs,
                candidate_k=k,
            )
            output[record.claim_id] = reranked
            latencies.append(elapsed_ms)
        mean = sum(latencies) / len(latencies) if latencies else 0.0
        return output, {"mean_query_latency_ms": mean}


def cached_scores_for_ranking(
    ranking: Ranking,
    claim_id: int,
    cache: Mapping[str, float],
) -> list[float]:
    keys = [f"{claim_id}:{doc_id}" for doc_id, _ in ranking]
    missing = [key for key in keys if key not in cache]
    if missing:
        raise ValueError(f"cross-encoder cache is missing {missing[0]}")
    return [float(cache[key]) for key in keys]


def blend_cached_rerank(
    ranking: Ranking,
    claim_id: int,
    zero_shot: Mapping[str, float],
    fine_tuned: Mapping[str, float] | None,
    *,
    k: int,
    lam: float,
    mode: str,
) -> list[tuple[int, float]]:
    prefix = list(ranking[:k])
    ce = cached_scores_for_ranking(prefix, claim_id, zero_shot)
    if mode == "zero_shot":
        scores = _zscore(ce)
    elif mode == "fine_tuned":
        if fine_tuned is None:
            raise ValueError("fine-tuned cache is required")
        scores = _zscore(cached_scores_for_ranking(prefix, claim_id, fine_tuned))
    elif mode == "mean":
        if fine_tuned is None:
            raise ValueError("fine-tuned cache is required")
        scores = [
            (left + right) / 2
            for left, right in zip(
                _zscore(ce),
                _zscore(cached_scores_for_ranking(prefix, claim_id, fine_tuned)),
            )
        ]
    else:
        raise ValueError(f"unsupported CE mode: {mode}")
    first_stage = _zscore([float(score) for _, score in prefix])
    reranked = sorted(
        (
            (
                doc_id,
                lam * scores[index] + (1 - lam) * first_stage[index],
            )
            for index, (doc_id, _) in enumerate(prefix)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    prefix_ids = {doc_id for doc_id, _ in prefix}
    reranked.extend(
        (doc_id, score) for doc_id, score in ranking[k:] if doc_id not in prefix_ids
    )
    return reranked


def select_candidate_k(
    validation_runs: Mapping[int, Mapping[int, Ranking]],
    validation_claims: Claims,
    corpus: Corpus,
    reranker: CrossEncoderReranker,
    candidates: Sequence[int],
) -> tuple[int, dict[int, float]]:
    """Select candidate k by validation nDCG@10."""
    if not candidates:
        raise ValueError("candidate k values must not be empty")
    gold = gold_documents(validation_claims)
    scores: dict[int, float] = {}
    for candidate_k in candidates:
        reranked, _ = reranker.rerank(
            validation_claims,
            validation_runs,
            corpus,
            candidate_k=candidate_k,
        )
        metrics = evaluate_retrieval(
            {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in reranked.items()},
            gold,
        )
        scores[candidate_k] = float(metrics["mean"]["ndcg@10"])
    best = max(scores, key=lambda candidate: (scores[candidate], -candidate))
    return best, scores


def _zscore(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    return [(value - mean) / scale for value in values] if scale else [0.0] * len(values)


def blend_rerank(
    ranking: Ranking,
    ce_scores: Sequence[float],
    *,
    k: int,
    lam: float,
) -> list[tuple[int, float]]:
    """Blend z-scored CE and first-stage scores, preserving the fill order."""
    if not 0 <= lam <= 1:
        raise ValueError("lambda must be between 0 and 1")
    prefix = list(ranking[:k])
    if len(ce_scores) != len(prefix):
        raise ValueError("cross-encoder score count does not match rerank prefix")
    ce_z = _zscore([float(score) for score in ce_scores])
    first_z = _zscore([float(score) for _, score in prefix])
    reranked = sorted(
        (
            (doc_id, lam * ce_z[index] + (1 - lam) * first_z[index])
            for index, (doc_id, _) in enumerate(prefix)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    prefix_ids = {doc_id for doc_id, _ in prefix}
    reranked.extend((doc_id, score) for doc_id, score in ranking[k:] if doc_id not in prefix_ids)
    return reranked


def run_task3(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    retrieval_runs: Mapping[str, Mapping[str, Mapping[int, Ranking]]] | None = None,
) -> dict[str, Any]:
    """Run configured fusion and cross-encoder reranking explicitly."""
    config = load_config(config_path)
    result_path = Path(config["paths"]["results_dir"]) / "task3_hybrid_rerank.json"
    prediction_dir = Path(config["paths"]["predictions_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    if result_path.is_file() and all(
        (prediction_dir / name).is_file()
        for name in ("t3_hybrid.trec", "t3_rerank.trec")
    ) and (results_dir / "val_rerank.trec").is_file() and not (
        Path(config["paths"]["cache_dir"]) / "ce_ft" / "ce_val.json"
    ).is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    set_global_seed(int(config["seed"]))
    corpus = load_corpus(config["paths"]["corpus"])
    splits = load_all_claim_splits(config["paths"]["data_dir"])
    candidate_depth = int(config["fusion"]["candidate_depth"])
    if retrieval_runs is None:
        dense_module = importlib.import_module("scripts.dense")
        retrieve_module = importlib.import_module("scripts.retrieve")
        bm25_config = config["bm25"]
        bm25_settings = retrieve_module.BM25Settings(
            k1=float(bm25_config["k1"]),
            b=float(bm25_config["b"]),
            stem=bool(bm25_config["stem"]),
            stopwords=bool(bm25_config["stopwords"]),
            fields=tuple(bm25_config["fields"]),
            depth=candidate_depth,
        )
        bm25_run_val = retrieve_module.BM25Retriever(corpus, bm25_settings).retrieve(
            splits["val"], depth=candidate_depth
        )
        bm25_run_eval = retrieve_module.BM25Retriever(corpus, bm25_settings).retrieve(
            splits["eval"], depth=candidate_depth
        )
        general_settings = dense_module._settings_from_config(config, "general")
        general_settings = dense_module.DenseSettings(
            **{**general_settings.__dict__, "depth": candidate_depth}
        )
        general_retriever = dense_module.DenseRetriever(corpus, general_settings)
        general_run_val, _ = general_retriever.retrieve(splits["val"], depth=candidate_depth)
        general_run_eval, _ = general_retriever.retrieve(splits["eval"], depth=candidate_depth)
    else:
        try:
            bm25_run_val = retrieval_runs["validation"]["bm25"]
            bm25_run_eval = retrieval_runs["evaluation"]["bm25"]
            general_run_val = retrieval_runs["validation"]["dense_general"]
            general_run_eval = retrieval_runs["evaluation"]["dense_general"]
        except KeyError as exc:
            raise ValueError(
                "retrieval_runs must contain validation/evaluation bm25 and dense_general runs"
            ) from exc

    scientific_run_val = scientific_run_eval = None
    if retrieval_runs is not None:
        scientific_run_val = retrieval_runs["validation"].get("dense_scientific")
        scientific_run_eval = retrieval_runs["evaluation"].get("dense_scientific")
    elif bool(config["fusion"].get("include_scientific", False)):
        dense_module = importlib.import_module("scripts.dense")
        scientific_settings = dense_module._settings_from_config(config, "scientific")
        scientific_settings = dense_module.DenseSettings(
            **{**scientific_settings.__dict__, "depth": candidate_depth}
        )
        scientific_retriever = dense_module.DenseRetriever(corpus, scientific_settings)
        scientific_run_val, _ = scientific_retriever.retrieve(
            splits["val"], depth=candidate_depth
        )
        scientific_run_eval, _ = scientific_retriever.retrieve(
            splits["eval"], depth=candidate_depth
        )

    gold = gold_documents(splits["val"])
    constants = [int(value) for value in config["fusion"].get("rrf_constants", [10, 30, 60])]
    candidates = [[bm25_run_val, general_run_val]]
    if scientific_run_val is not None:
        candidates.append([bm25_run_val, general_run_val, scientific_run_val])
    fusion_scores: dict[str, dict[int, float]] = {}
    best_fusion: tuple[str, int] | None = None
    best_score = -1.0
    for index, runs in enumerate(candidates):
        constant, scores = compare_rrf_constants(runs, gold, constants, depth=candidate_depth)
        key = "bm25_general" if index == 0 else "bm25_general_scientific"
        fusion_scores[key] = scores
        score = scores[constant]
        if score > best_score:
            best_score = score
            best_fusion = (key, constant)
    if best_fusion is None:
        raise ValueError("no fusion candidates available")
    fusion_runs = candidates[0] if best_fusion[0] == "bm25_general" else candidates[1]
    hybrid_val = fuse_runs(*fusion_runs[:2], scientific_dense_run=fusion_runs[2] if len(fusion_runs) == 3 else None, method="rrf", depth=candidate_depth, rrf_constant=best_fusion[1])
    hybrid_eval = fuse_runs(bm25_run_eval, general_run_eval, scientific_dense_run=scientific_run_eval if len(fusion_runs) == 3 else None, method="rrf", depth=candidate_depth, rrf_constant=best_fusion[1])
    fusion_config = config["fusion"]
    hybrid_search_config = fusion_config["hybrid_search"]
    tuned_score, tuned_spec = search_hybrid(
        bm25_run_val,
        general_run_val,
        gold,
        rrf_constants=[int(value) for value in fusion_config["rrf_constants"]],
        dense_weights=[
            float(value) for value in hybrid_search_config["dense_weights"]
        ],
        interpolation_alphas=[
            float(value)
            for value in hybrid_search_config["interpolation_alphas"]
        ],
        depth=candidate_depth,
    )
    hybrid_spec: dict[str, Any] = {}
    if tuned_score > best_score:
        hybrid_val = apply_hybrid(
            tuned_spec, bm25_run_val, general_run_val, depth=candidate_depth
        )
        hybrid_eval = apply_hybrid(
            tuned_spec, bm25_run_eval, general_run_eval, depth=candidate_depth
        )
        hybrid_spec = {"runs": "bm25_general", **tuned_spec}
    first_stage = {
        "bm25": bm25_run_val,
        "dense_general": general_run_val,
        "hybrid": hybrid_val,
    }
    first_stage_eval = {
        "bm25": bm25_run_eval,
        "dense_general": general_run_eval,
        "hybrid": hybrid_eval,
    }
    if scientific_run_val is not None:
        first_stage["dense_scientific"] = scientific_run_val
        first_stage_eval["dense_scientific"] = scientific_run_eval
    first_stage_metrics = {
        name: evaluate_retrieval(
            {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in run.items()},
            gold,
        )["mean"]
        for name, run in first_stage.items()
    }
    best_first_stage = max(
        first_stage_metrics,
        key=lambda name: (
            float(first_stage_metrics[name]["ndcg@10"]),
            -list(first_stage).index(name),
        ),
    )
    first_stage_val = first_stage[best_first_stage]
    rerank_config = config["rerank"]
    reranker = CrossEncoderReranker(
        RerankSettings(
            name=str(rerank_config["model_name"]),
            batch_size=int(rerank_config["batch_size"]),
            device=resolve_device(str(rerank_config["device"])),
            candidate_k=int(rerank_config["candidate_k"]),
            max_k=int(rerank_config["max_k"]),
            max_length=int(rerank_config["max_length"]),
        )
    )
    cache_dir = Path(config["paths"]["cache_dir"])
    val_ce, val_latency_ms = reranker.score_cache(
        splits["val"], first_stage_val, corpus, cache_dir / "ce_val.json"
    )
    eval_ce, eval_latency_ms = reranker.score_cache(
        splits["eval"], first_stage_eval[best_first_stage], corpus, cache_dir / "ce_eval.json"
    )
    ft_val_path = cache_dir / "ce_ft" / "ce_val.json"
    ft_eval_path = cache_dir / "ce_ft" / "ce_eval.json"
    ft_val = (
        {str(key): float(value) for key, value in json.loads(ft_val_path.read_text()).items()}
        if ft_val_path.is_file() else None
    )
    ft_eval = (
        {str(key): float(value) for key, value in json.loads(ft_eval_path.read_text()).items()}
        if ft_eval_path.is_file() else None
    )
    k_grid = [int(value) for value in rerank_config["candidate_grid"]]
    lambdas = [float(value) for value in rerank_config["lambda_grid"]]
    blend_scores: dict[str, dict[str, float]] = {}
    for mode in ("zero_shot", "fine_tuned", "mean"):
        if mode != "zero_shot" and ft_val is None:
            continue
        for k in k_grid:
            for lam in lambdas:
                run = {
                    record.claim_id: blend_cached_rerank(
                        first_stage_val[record.claim_id],
                        record.claim_id,
                        val_ce,
                        ft_val,
                        k=k,
                        lam=lam,
                        mode=mode,
                    )
                    for record in splits["val"].records
                }
                blend_scores.setdefault(f"{k}:{lam}", {})[mode] = float(
                    evaluate_retrieval(
                        {cid: [doc_id for doc_id, _ in ranking] for cid, ranking in run.items()},
                        gold,
                    )["mean"]["ndcg@10"]
                )
    selected_key, selected_mode = max(
        (
            (key, mode)
            for key, scores in blend_scores.items()
            for mode in scores
        ),
        key=lambda item: (
            blend_scores[item[0]][item[1]],
            -int(item[0].split(":")[0]),
            item[1],
        ),
    )
    selected_k, selected_lam = (
        int(selected_key.split(":")[0]),
        float(selected_key.split(":")[1]),
    )
    rerank_val = {
        record.claim_id: blend_cached_rerank(
            first_stage_val[record.claim_id],
            record.claim_id,
            val_ce,
            ft_val,
            k=selected_k,
            lam=selected_lam,
            mode=selected_mode,
        )
        for record in splits["val"].records
    }
    rerank_eval = {
        record.claim_id: blend_cached_rerank(
            first_stage_eval[best_first_stage][record.claim_id],
            record.claim_id,
            eval_ce,
            ft_eval,
            k=selected_k,
            lam=selected_lam,
            mode=selected_mode,
        )
        for record in splits["eval"].records
    }
    prediction_dir = Path(config["paths"]["predictions_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    expected_eval = {record.claim_id for record in splits["eval"].records}
    write_trec(prediction_dir / "t3_hybrid.trec", {
        cid: ranking[:100] for cid, ranking in hybrid_eval.items()
    }, "hybrid", expected_claim_ids=expected_eval)
    write_trec(prediction_dir / "t3_rerank.trec", {
        cid: ranking[:100] for cid, ranking in rerank_eval.items()
    }, "rerank", expected_claim_ids=expected_eval)
    write_trec(
        results_dir / "val_rerank.trec",
        {cid: ranking[:100] for cid, ranking in rerank_val.items()},
        "rerank",
        expected_claim_ids={record.claim_id for record in splits["val"].records},
    )
    result = {
        "fusion": {
            "selected": hybrid_spec or {"runs": best_fusion[0], "rrf_constant": best_fusion[1]},
            "grid_scores": fusion_scores,
            "first_stage_metrics": first_stage_metrics,
            "selected_first_stage": best_first_stage,
        },
        "rerank": {
            "model_name": rerank_config["model_name"],
            "parameter_count": reranker.parameter_count,
            "selected_k": selected_k,
            "selected_lambda": selected_lam,
            "selected_mode": selected_mode,
            "blend_scores": blend_scores,
            "validation_metrics": evaluate_retrieval(
                {cid: [doc_id for doc_id, _ in ranking] for cid, ranking in rerank_val.items()},
                gold,
            )["mean"],
            "validation_latency": {"mean_query_latency_ms": val_latency_ms},
            "evaluation_latency": {"mean_query_latency_ms": eval_latency_ms},
        },
    }
    metrics = {
        name: first_stage_metrics.get(name, {}).get("ndcg@10")
        for name in ("bm25", "dense_general", "hybrid")
    }
    metrics["rerank"] = float(evaluate_retrieval(
        {cid: [doc_id for doc_id, _ in ranking] for cid, ranking in rerank_val.items()}, gold
    )["mean"]["ndcg@10"])
    summary = {
        "val_ndcg10": metrics,
        "per_query_latency_ms": {
            "bm25": None, "dense": None, "hybrid": None,
            "rerank": val_latency_ms,
        },
        "ndcg_gain_over_previous": {
            "bm25": None,
            "dense": None if metrics["dense_general"] is None else metrics["dense_general"] - metrics["bm25"],
            "hybrid": metrics["hybrid"] - metrics["dense_general"],
            "rerank": metrics["rerank"] - float(first_stage_metrics[best_first_stage]["ndcg@10"]),
        },
        "rerank_selection_val_ndcg10": {
            (
                "fine_tuned_ce_extra_not_required_by_ps"
                if mode == "fine_tuned"
                else mode
            ): max(scores[mode] for scores in blend_scores.values() if mode in scores)
            for mode in ("zero_shot", "fine_tuned", "mean")
            if any(mode in scores for scores in blend_scores.values())
        },
    }
    result_path = Path(config["paths"]["results_dir"]) / "task3_hybrid_rerank.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (result_path.parent / "task3_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run_task3()
