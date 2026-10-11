"""Per-query latency measurements for the retrieval pipeline."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

from .common import DEFAULT_CONFIG, load_config, resolve_device, set_global_seed
from .data import load_all_claim_splits, load_corpus
from .dense import DenseRetriever, _settings_from_config
from .fuse import fuse_runs
from .rerank import CrossEncoderReranker, RerankSettings, document_text
from .retrieve import BM25Retriever, BM25Settings


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("latency measurement produced no queries")
    return sum(values) / len(values)


def measure_latency(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    set_global_seed(int(config["seed"]))
    corpus = load_corpus(config["paths"]["corpus"])
    claims = load_all_claim_splits(config["paths"]["data_dir"])["val"].records[:50]
    claim_subset = type(load_all_claim_splits(config["paths"]["data_dir"])["val"])(
        list(claims),
        {record.claim_id: index for index, record in enumerate(claims)},
    )
    selected = json.loads(
        (Path(config["paths"]["results_dir"]) / "task1_bm25_tuning.json").read_text()
    )["selected_settings"]
    bm25 = BM25Retriever(corpus, BM25Settings(
        k1=float(selected["k1"]), b=float(selected["b"]),
        stem=bool(selected["stem"]), stopwords=bool(selected["stopwords"]),
        fields=tuple(selected["fields"]), depth=100,
    ))
    warmup_queries = 2
    general = DenseRetriever(corpus, _settings_from_config(config, "general"))
    scientific = DenseRetriever(corpus, _settings_from_config(config, "scientific"))
    bm25_times: list[float] = []
    general_times: list[float] = []
    scientific_times: list[float] = []
    bm25_runs: dict[int, list[tuple[int, float]]] = {}
    general_runs: dict[int, list[tuple[int, float]]] = {}
    scientific_runs: dict[int, list[tuple[int, float]]] = {}
    for query_index, record in enumerate(claim_subset.records):
        start = time.perf_counter()
        bm25_runs[record.claim_id] = bm25.retrieve(
            type(claim_subset)([record], {record.claim_id: 0}), depth=100
        )[record.claim_id]
        if query_index >= warmup_queries:
            bm25_times.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        general_runs[record.claim_id] = general.retrieve(
            type(claim_subset)([record], {record.claim_id: 0}), depth=100
        )[0][record.claim_id]
        if query_index >= warmup_queries:
            general_times.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        scientific_runs[record.claim_id] = scientific.retrieve(
            type(claim_subset)([record], {record.claim_id: 0}), depth=100
        )[0][record.claim_id]
        if query_index >= warmup_queries:
            scientific_times.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    hybrid = fuse_runs(bm25_runs, general_runs, scientific_dense_run=scientific_runs, depth=200)
    hybrid_time = (time.perf_counter() - start) * 1000 / len(claim_subset.records)
    reranker = CrossEncoderReranker(RerankSettings(
        name=str(config["rerank"]["model_name"]),
        batch_size=int(config["rerank"]["batch_size"]),
        device=resolve_device(str(config["rerank"]["device"])),
        candidate_k=int(config["rerank"]["max_k"]),
        max_k=int(config["rerank"]["max_k"]),
        max_length=int(config["rerank"]["max_length"]),
    ))
    rerank_times: list[float] = []
    for query_index, record in enumerate(claim_subset.records):
        pairs = [
            (record.claim, document_text(corpus, doc_id))
            for doc_id, _ in hybrid[record.claim_id][: int(config["rerank"]["max_k"])]
        ]
        start = time.perf_counter()
        reranker.score_pairs(pairs)
        if query_index >= warmup_queries:
            rerank_times.append((time.perf_counter() - start) * 1000)
    try:
        import torch
        hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
        cuda_peak_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else None
        )
    except ImportError:
        hardware = platform.processor()
        cuda_peak_memory_mb = None
    result = {
        "queries": len(claim_subset.records),
        "hardware": hardware or platform.machine(),
        "bm25_ms_per_query": _mean(bm25_times),
        "general_dense_ms_per_query": _mean(general_times),
        "scientific_dense_ms_per_query": _mean(scientific_times),
        "hybrid_ms_per_query": hybrid_time,
        "rerank_ms_per_query": _mean(rerank_times),
        "warmup_queries": warmup_queries,
        "cuda_peak_memory_mb": cuda_peak_memory_mb,
    }
    path = Path(config["paths"]["results_dir"]) / "latency.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
