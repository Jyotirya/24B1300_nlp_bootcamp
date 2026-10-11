"""BM25 lexical retrieval for SciFact Task 1."""

from __future__ import annotations

import json
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import DEFAULT_CONFIG, load_config, set_global_seed
from .data import Claims, Corpus, load_all_claim_splits, load_corpus
from .metrics import evaluate_retrieval, gold_documents, write_trec

TOKEN_PATTERN = re.compile(r"(?u)\b[\w]+(?:-[\w]+)*\b")
DEFAULT_STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or that the this to was with".split()
)


@dataclass(frozen=True)
class BM25Settings:
    k1: float = 1.2
    b: float = 0.75
    stem: bool = False
    stopwords: bool = True
    fields: tuple[str, ...] = ("title", "abstract")
    depth: int = 100


def tokenize_text(
    text: str,
    *,
    stem: bool = False,
    remove_stopwords: bool = True,
) -> list[str]:
    """Tokenize scientific text while retaining terms such as ``IL-6`` and ``p53``."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
    if remove_stopwords:
        tokens = [token for token in tokens if token not in DEFAULT_STOPWORDS]
    if stem:
        try:
            stemmer_module = importlib.import_module("Stemmer")
        except ImportError as exc:
            raise ImportError(
                "PyStemmer is required when BM25 stemming is enabled"
            ) from exc
        stemmer = stemmer_module.Stemmer("english")
        tokens = stemmer.stemWords(tokens)
    return tokens


def _record_text(record: Any, fields: Sequence[str]) -> str:
    parts: list[str] = []
    for field in fields:
        if field == "title":
            parts.append(record.title)
        elif field == "abstract":
            parts.extend(record.abstract)
        else:
            raise ValueError(f"Unsupported corpus field: {field}")
    return " ".join(parts)


def _bm25_tokenize(
    texts: Sequence[str],
    *,
    stem: bool,
    remove_stopwords: bool,
) -> Any:
    """Use bm25s tokenization with the project tokenizer's scientific pattern."""
    try:
        bm25s = importlib.import_module("bm25s")
    except ImportError as exc:
        raise ImportError(
            "bm25s is required for BM25 retrieval; install code/scripts/requirements.txt"
        ) from exc
    stemmer = None
    if stem:
        try:
            stemmer_module = importlib.import_module("Stemmer")
        except ImportError as exc:
            raise ImportError(
                "PyStemmer is required when BM25 stemming is enabled"
            ) from exc
        stemmer = stemmer_module.Stemmer("english")
    stopwords: str | list[str] | None = list(DEFAULT_STOPWORDS) if remove_stopwords else None
    return bm25s.tokenize(
        list(texts),
        token_pattern=TOKEN_PATTERN.pattern,
        stopwords=stopwords,
        stemmer=stemmer,
        show_progress=False,
    )


class BM25Retriever:
    """A bm25s-backed retriever that maps internal indexes to corpus IDs."""

    def __init__(
        self,
        corpus: Corpus,
        settings: BM25Settings,
        *,
        tokenized_corpus: Any | None = None,
    ) -> None:
        if settings.k1 <= 0 or settings.b < 0 or settings.b > 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        if settings.depth < 100:
            raise ValueError("BM25 depth must be at least 100")
        self.corpus = corpus
        self.settings = settings
        self._texts = [
            _record_text(record, settings.fields) for record in corpus.records
        ]
        try:
            bm25s = importlib.import_module("bm25s")
        except ImportError as exc:
            raise ImportError(
                "bm25s is required for BM25 retrieval; install code/scripts/requirements.txt"
            ) from exc
        self._backend = bm25s.BM25(method="lucene", k1=settings.k1, b=settings.b)
        if tokenized_corpus is None:
            tokenized_corpus = _bm25_tokenize(
                self._texts,
                stem=settings.stem,
                remove_stopwords=settings.stopwords,
            )
        self._backend.index(tokenized_corpus, show_progress=False)

    def _rank_tokens(self, query_tokens: Any, requested_depth: int) -> list[tuple[int, float]]:
        indexes, scores = self._backend.retrieve(
            query_tokens,
            k=requested_depth,
            show_progress=False,
        )
        raw_indexes = indexes[0].tolist()
        raw_scores = scores[0].tolist()
        ranked = rank_document_indices(raw_indexes, raw_scores, self.corpus)
        seen = {doc_id for doc_id, _ in ranked}
        if len(ranked) < len(self.corpus.records):
            ranked.extend(
                (record.doc_id, 0.0)
                for record in self.corpus.records
                if record.doc_id not in seen
            )
        return ranked[: min(requested_depth, len(ranked))]

    def rank_claim(self, claim: str, *, depth: int | None = None) -> list[tuple[int, float]]:
        """Return a complete corpus ranking truncated to ``depth``."""
        requested_depth = self.settings.depth if depth is None else depth
        if requested_depth < 100:
            raise ValueError("retrieval depth must be at least 100")
        if not self.corpus.records:
            return []
        query_tokens = _bm25_tokenize(
            [claim],
            stem=self.settings.stem,
            remove_stopwords=self.settings.stopwords,
        )
        return self._rank_tokens(query_tokens, requested_depth)

    def retrieve(self, claims: Claims, *, depth: int | None = None) -> dict[int, list[tuple[int, float]]]:
        """Retrieve independently for every claim without using evidence."""
        requested_depth = self.settings.depth if depth is None else depth
        if requested_depth < 100:
            raise ValueError("retrieval depth must be at least 100")
        query_tokens = _bm25_tokenize(
            [record.claim for record in claims.records],
            stem=self.settings.stem,
            remove_stopwords=self.settings.stopwords,
        )
        indexes, scores = self._backend.retrieve(
            query_tokens,
            k=requested_depth,
            show_progress=False,
        )
        rankings: dict[int, list[tuple[int, float]]] = {}
        for index, record in enumerate(claims.records):
            ranked = rank_document_indices(
                indexes[index].tolist(),
                scores[index].tolist(),
                self.corpus,
            )
            seen = {doc_id for doc_id, _ in ranked}
            if len(ranked) < len(self.corpus.records):
                ranked.extend(
                    (corpus_record.doc_id, 0.0)
                    for corpus_record in self.corpus.records
                    if corpus_record.doc_id not in seen
                )
            rankings[record.claim_id] = ranked[: min(requested_depth, len(ranked))]
        return rankings


def rank_document_indices(
    indexes: Sequence[int],
    scores: Sequence[float],
    corpus: Corpus,
) -> list[tuple[int, float]]:
    """Convert backend indexes to integer IDs with deterministic score ordering."""
    if len(indexes) != len(scores):
        raise ValueError("backend indexes and scores must have equal lengths")
    pairs = [(int(index), float(score)) for index, score in zip(indexes, scores)]
    if any(index < 0 or index >= len(corpus.records) for index, _ in pairs):
        raise ValueError("backend returned an index outside the corpus")
    pairs.sort(key=lambda item: (-item[1], item[0]))
    seen: set[int] = set()
    unique_pairs: list[tuple[int, float]] = []
    for index, score in pairs:
        if index not in seen:
            unique_pairs.append((index, score))
            seen.add(index)
    return [(corpus.records[index].doc_id, score) for index, score in unique_pairs]


def _settings_grid(config: Mapping[str, Any]) -> list[BM25Settings]:
    grid = config.get("bm25", {}).get("grid", {})
    return [
        BM25Settings(
            k1=float(k1),
            b=float(b),
            stem=bool(stem),
            stopwords=bool(config.get("bm25", {}).get("stopwords", True)),
            fields=tuple(config.get("bm25", {}).get("fields", ["title", "abstract"])),
            depth=int(config.get("bm25", {}).get("depth", 100)),
        )
        for k1 in grid.get("k1", [1.2])
        for b in grid.get("b", [0.75])
        for stem in grid.get("stem", [False])
    ]


def tune_bm25(
    corpus: Corpus,
    validation_claims: Claims,
    settings_grid: Iterable[BM25Settings],
    output_path: str | Path,
    *,
    ndcg_k: int = 10,
    recall_k: int = 100,
) -> dict[str, Any]:
    """Evaluate every BM25 setting on validation nDCG and save all results."""
    gold = gold_documents(validation_claims)
    grid_results: list[dict[str, Any]] = []
    token_cache: dict[bool, Any] = {}
    for settings in settings_grid:
        if settings.stem not in token_cache:
            token_cache[settings.stem] = _bm25_tokenize(
                [_record_text(record, settings.fields) for record in corpus.records],
                stem=settings.stem,
                remove_stopwords=settings.stopwords,
            )
        retriever = BM25Retriever(
            corpus,
            settings,
            tokenized_corpus=token_cache[settings.stem],
        )
        run = retriever.retrieve(validation_claims)
        metrics = evaluate_retrieval(
            {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in run.items()},
            gold,
            ndcg_k=ndcg_k,
            recall_k=recall_k,
        )
        grid_results.append(
            {
                "settings": {
                    "k1": settings.k1,
                    "b": settings.b,
                    "stem": settings.stem,
                    "stopwords": settings.stopwords,
                    "fields": list(settings.fields),
                    "depth": settings.depth,
                },
                "metrics": metrics["mean"],
            }
        )
    if not grid_results:
        raise ValueError("BM25 settings grid must not be empty")
    best = max(
        grid_results,
        key=lambda result: (
            float(result["metrics"]["ndcg@10"]),
            float(result["metrics"]["recall@100"]),
            float(result["metrics"]["mrr"]),
        ),
    )
    result = {
        "selected_settings": best["settings"],
        "best_validation_metrics": best["metrics"],
        "grid_results": grid_results,
    }
    result_file = Path(output_path)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_task1(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Tune BM25 and write Task 1 validation/evaluation runs when explicitly called."""
    config = load_config(config_path)
    result_path = Path(config["paths"]["results_dir"]) / "task1_bm25_tuning.json"
    prediction_dir = Path(config["paths"]["predictions_dir"])
    if result_path.is_file() and all(
        (prediction_dir / name).is_file()
        for name in ("t1_bm25.trec", "val_bm25.trec")
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    set_global_seed(int(config["seed"]))
    corpus = load_corpus(config["paths"]["corpus"])
    splits = load_all_claim_splits(config["paths"]["data_dir"])
    tuning = tune_bm25(
        corpus,
        splits["val"],
        _settings_grid(config),
        result_path,
        ndcg_k=int(config["evaluation"]["ndcg_k"]),
        recall_k=int(config["evaluation"]["recall_k"]),
    )
    selected = BM25Settings(**{
        key: value for key, value in tuning["selected_settings"].items()
    })
    retriever = BM25Retriever(corpus, selected)
    train_run = retriever.retrieve(splits["train"], depth=selected.depth)
    cache_dir = Path(config["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "task1_bm25_train.json").write_text(
        json.dumps(train_run) + "\n", encoding="utf-8"
    )
    for split, filename in (("eval", "t1_bm25.trec"), ("val", "val_bm25.trec")):
        run = retriever.retrieve(splits[split], depth=selected.depth)
        write_trec(
            Path(config["paths"]["predictions_dir"]) / filename,
            run,
            "bm25",
            expected_claim_ids={record.claim_id for record in splits[split].records},
            top_k=int(config["evaluation"]["trec_top_k"]),
        )
    return tuning


if __name__ == "__main__":
    run_task1()
