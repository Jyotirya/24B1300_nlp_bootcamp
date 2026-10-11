"""Sentence-transformer dense retrieval for SciFact Task 2."""

from __future__ import annotations

import importlib
import json
import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .common import DEFAULT_CONFIG, load_config, resolve_device, set_global_seed
from .data import Claims, Corpus, load_all_claim_splits, load_corpus
from .metrics import evaluate_retrieval, gold_documents, write_trec
from .timing import latency_statistics, process_rss_mb

MAX_MODEL_PARAMETERS = 400_000_000
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class DenseSettings:
    name: str
    query_prefix: str = ""
    document_prefix: str = ""
    max_seq_length: int = 512
    batch_size: int = 32
    device: str = "auto"
    depth: int = 100
    warmup_queries: int = 1
    pooling: str = "sentence_transformer"
    cache_dir: str = "cache"
    results_dir: str = "results"


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return row-normalized embeddings, rejecting zero-norm rows."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embeddings contain a zero-norm row")
    return values / norms


def count_model_parameters(model: Any) -> int:
    """Count actual model parameters and enforce the bootcamp limit."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise TypeError("loaded model does not expose parameters()")
    count = sum(int(parameter.numel()) for parameter in parameters())
    if count > MAX_MODEL_PARAMETERS:
        raise ValueError(
            f"model has {count:,} parameters, exceeding the "
            f"{MAX_MODEL_PARAMETERS:,}-parameter limit"
        )
    return count


def apply_query_prefix(query: str, settings: DenseSettings) -> str:
    """Apply the configured query prefix exactly once."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    prefix = settings.query_prefix
    if not prefix and settings.name == "BAAI/bge-base-en-v1.5":
        prefix = BGE_QUERY_PREFIX
    return f"{prefix}{query}"


def _document_text(record: Any) -> str:
    return f"{record.title}\n{' '.join(record.abstract)}"


def rank_embeddings(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    corpus: Corpus,
    depth: int,
) -> list[tuple[int, float]]:
    """Rank normalized query/document embeddings with deterministic ties."""
    if depth < 100:
        raise ValueError("dense retrieval depth must be at least 100")
    query = normalize_embeddings(np.asarray(query_embedding).reshape(1, -1))[0]
    documents = np.asarray(document_embeddings)
    if documents.ndim != 2 or documents.shape[0] != len(corpus.records):
        raise ValueError("document embeddings must align with corpus records")
    scores = documents @ query
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    return [
        (corpus.records[index].doc_id, float(scores[index]))
        for index in order[: min(depth, len(order))]
    ]


class DenseRetriever:
    """A sentence-transformer bi-encoder with precomputed document embeddings."""

    def __init__(self, corpus: Corpus, settings: DenseSettings) -> None:
        if settings.depth < 100:
            raise ValueError("dense retrieval depth must be at least 100")
        if settings.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if settings.device == "cpu":
            torch = importlib.import_module("torch")
            torch.set_num_threads(os.cpu_count() or 1)
        self.corpus = corpus
        self.settings = DenseSettings(**{
            **settings.__dict__,
            "device": resolve_device(settings.device),
        })
        settings = self.settings
        self._scientific = settings.pooling == "cls"
        rss_before = process_rss_mb()
        if self._scientific:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
            self._torch = torch
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(settings.name)
            self.model = transformers.AutoModel.from_pretrained(settings.name)
            self.model.to(settings.device)
            self.model.eval()
        else:
            sentence_transformers = importlib.import_module("sentence_transformers")
            self.model = sentence_transformers.SentenceTransformer(
                settings.name,
                device=settings.device,
            )
        count_model_parameters(self.model)
        if not self._scientific:
            self.model.max_seq_length = settings.max_seq_length
        documents = [_document_text(record) for record in corpus.records]
        cache_path = Path(settings.cache_dir) / f"emb_{_cache_name(settings.name)}.npy"
        timing_path = Path(settings.results_dir) / "dense_timing.json"
        timing = _load_timing(timing_path)
        start = time.perf_counter()
        if cache_path.is_file():
            self.document_embeddings = np.load(cache_path)
            self.indexing_time_sec = float(
                timing.get(settings.name, {}).get("index_time_sec", 0.0)
            )
        else:
            encoded = self._encode_documents(documents)
            self.indexing_time_sec = time.perf_counter() - start
            self.document_embeddings = normalize_embeddings(np.asarray(encoded))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.document_embeddings)
            timing[settings.name] = {
                "index_time_sec": self.indexing_time_sec,
            }
            timing_path.parent.mkdir(parents=True, exist_ok=True)
            timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
        self.memory_delta_mb = process_rss_mb() - rss_before
        if self.document_embeddings.shape[0] != len(corpus.records):
            raise ValueError("model returned an embedding for every corpus record")

    def _encode_documents(self, documents: list[str]) -> np.ndarray:
        if not self._scientific:
            return np.asarray(self.model.encode(
                [f"{self.settings.document_prefix}{text}" for text in documents],
                batch_size=self.settings.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ))
        lengths = sorted(
            range(len(documents)),
            key=lambda index: len(documents[index]),
        )
        encoded = [None] * len(documents)
        separator = self._tokenizer.sep_token or "[SEP]"
        documents = [
            text.replace("\n", f" {separator} ", 1) for text in documents
        ]
        for start in range(0, len(lengths), self.settings.batch_size):
            batch_indices = lengths[start : start + self.settings.batch_size]
            encoded_batch = self._encode_transformer(
                [documents[index] for index in batch_indices]
            )
            for index, vector in zip(batch_indices, encoded_batch):
                encoded[index] = vector
        return np.asarray(encoded)

    def _encode_transformer(self, texts: list[str]) -> np.ndarray:
        tokens = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.settings.max_seq_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.settings.device) for key, value in tokens.items()}
        with self._torch.no_grad():
            hidden = self.model(**tokens).last_hidden_state[:, 0]
        return hidden.detach().cpu().numpy()

    def _encode_query(self, claim: str) -> np.ndarray:
        if not self._scientific:
            return normalize_embeddings(np.asarray(self.model.encode(
                [apply_query_prefix(claim, self.settings)],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )))
        return normalize_embeddings(self._encode_transformer([claim]))

    @property
    def parameter_count(self) -> int:
        return count_model_parameters(self.model)

    def retrieve(
        self,
        claims: Claims,
        *,
        depth: int | None = None,
        measure_latency: bool = False,
    ) -> tuple[dict[int, list[tuple[int, float]]], dict[str, float] | None]:
        """Encode queries in batches and return rankings plus optional latency stats."""
        requested_depth = self.settings.depth if depth is None else depth
        if requested_depth < 100:
            raise ValueError("dense retrieval depth must be at least 100")
        rankings: dict[int, list[tuple[int, float]]] = {}
        latencies: list[float] = []
        for index, record in enumerate(claims.records):
            query_start = time.perf_counter()
            query_embedding = self._encode_query(record.claim)
            ranking = rank_embeddings(
                query_embedding,
                self.document_embeddings,
                self.corpus,
                requested_depth,
            )
            rankings[record.claim_id] = ranking
            if measure_latency and index >= self.settings.warmup_queries:
                latencies.append((time.perf_counter() - query_start) * 1000)
        return rankings, latency_statistics(latencies) if latencies else None


def _settings_from_config(config: Mapping[str, Any], model_key: str) -> DenseSettings:
    dense = config.get("dense", {})
    model = dense.get("models", {}).get(model_key)
    if not isinstance(model, dict):
        raise ValueError(f"Missing dense model configuration: {model_key}")
    return DenseSettings(
        name=str(model["name"]),
        query_prefix=str(model.get("query_prefix", "")),
        document_prefix=str(model.get("document_prefix", "")),
        max_seq_length=int(dense.get("max_seq_length", 512)),
        batch_size=int(dense.get("batch_size", 32)),
        device=resolve_device(str(dense.get("device", "auto"))),
        depth=int(dense.get("depth", 100)),
        warmup_queries=int(dense.get("warmup_queries", 1)),
        pooling=str(model.get("pooling", "sentence_transformer")),
        cache_dir=str(config["paths"]["cache_dir"]),
        results_dir=str(config["paths"]["results_dir"]),
    )


def _cache_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def _load_timing(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"dense timing file must contain an object: {path}")
    return value


def run_task2(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Run both configured dense models and write Task 2 artifacts explicitly."""
    config = load_config(config_path)
    result_path = Path(config["paths"]["results_dir"]) / "task2_dense.json"
    prediction_dir = Path(config["paths"]["predictions_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    if result_path.is_file() and all(
        (prediction_dir / name).is_file()
        for name in (
            "t2_general.trec",
            "t2_scientific.trec",
            "val_general.trec",
        )
    ) and (results_dir / "val_scientific.trec").is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    set_global_seed(int(config["seed"]))
    corpus = load_corpus(config["paths"]["corpus"])
    splits = load_all_claim_splits(config["paths"]["data_dir"])
    torch = importlib.import_module("torch")
    torch.set_num_threads(os.cpu_count() or 1)
    result: dict[str, Any] = {"models": {}}
    for model_key in ("general", "scientific"):
        settings = _settings_from_config(config, model_key)
        retriever = DenseRetriever(corpus, settings)
        val_run, latency = retriever.retrieve(
            splits["val"], measure_latency=True
        )
        val_rankings = {
            claim_id: [doc_id for doc_id, _ in ranking]
            for claim_id, ranking in val_run.items()
        }
        val_metrics = evaluate_retrieval(
            val_rankings,
            gold_documents(splits["val"]),
            ndcg_k=int(config["evaluation"]["ndcg_k"]),
            recall_k=int(config["evaluation"]["recall_k"]),
        )["mean"]
        eval_run, _ = retriever.retrieve(splits["eval"])
        if model_key == "general":
            train_run, _ = retriever.retrieve(splits["train"])
            cache_dir = Path(config["paths"]["cache_dir"])
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "task2_general_train.json").write_text(
                json.dumps(train_run) + "\n", encoding="utf-8"
            )
        prediction_dir = Path(config["paths"]["predictions_dir"])
        write_trec(
            prediction_dir / f"t2_{model_key}.trec",
            eval_run,
            f"dense_{model_key}",
            expected_claim_ids={record.claim_id for record in splits["eval"].records},
            top_k=int(config["evaluation"]["trec_top_k"]),
        )
        if model_key == "general":
            write_trec(
                prediction_dir / "val_general.trec",
                val_run,
                "dense_general",
                expected_claim_ids={record.claim_id for record in splits["val"].records},
                top_k=int(config["evaluation"]["trec_top_k"]),
            )
        else:
            write_trec(
                results_dir / "val_scientific.trec",
                val_run,
                "dense_scientific",
                expected_claim_ids={record.claim_id for record in splits["val"].records},
                top_k=int(config["evaluation"]["trec_top_k"]),
            )
        result["models"][model_key] = {
            "settings": settings.__dict__,
            "parameter_count": retriever.parameter_count,
            "indexing_time_sec": retriever.indexing_time_sec,
            "query_latency": latency,
            "memory_delta_mb": retriever.memory_delta_mb,
            "validation_metrics": val_metrics,
        }
        del retriever
        gc.collect()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run_task2()
