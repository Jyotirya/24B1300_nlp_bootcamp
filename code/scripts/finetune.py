"""Manual bi-encoder fine-tuning for SciFact Task 4."""

from __future__ import annotations

import copy
import gc
import importlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import DEFAULT_CONFIG, load_config, project_path, resolve_device, set_global_seed
from .data import Claims, Corpus, load_all_claim_splits, load_corpus
from .metrics import evaluate_retrieval, gold_documents, write_trec
from .dense import DenseRetriever, _cache_name, _settings_from_config
from .retrieve import BM25Retriever, BM25Settings


@dataclass(frozen=True)
class PositivePair:
    claim_id: int
    claim: str
    doc_id: int
    document: str


@dataclass(frozen=True)
class HardNegativeExample:
    claim_id: int
    claim: str
    positive_doc_id: int
    positive_document: str
    negative_doc_id: int
    negative_document: str


@dataclass(frozen=True)
class FineTuneSettings:
    base_model: str
    output_dir: str
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 16
    warmup_ratio: float = 0.1
    max_doc_length: int = 512
    max_query_length: int = 64
    device: str = "auto"
    depth: int = 100
    hard_negative_min_rank: int = 10
    hard_negative_max_rank: int = 50
    seed: int = 42
    query_prefix: str = ""
    pooling: str = "mean"


def document_text(corpus: Corpus, doc_id: int) -> str:
    index = corpus.id_to_index.get(doc_id)
    if index is None:
        raise ValueError(f"unknown document ID: {doc_id}")
    record = corpus.records[index]
    return f"{record.title}\n{' '.join(record.abstract)}"


def build_positive_pairs(train_claims: Claims, corpus: Corpus) -> list[PositivePair]:
    pairs: list[PositivePair] = []
    for record in train_claims.records:
        for evidence in record.evidence:
            pairs.append(
                PositivePair(
                    record.claim_id,
                    record.claim,
                    evidence.doc_id,
                    document_text(corpus, evidence.doc_id),
                )
            )
    return pairs


def mine_hard_negatives(
    train_claims: Claims,
    corpus: Corpus,
    bm25_run: Mapping[int, Sequence[tuple[int, float]]],
    dense_run: Mapping[int, Sequence[tuple[int, float]]],
    scientific_run: Mapping[int, Sequence[tuple[int, float]]] | None = None,
    *,
    min_rank: int = 10,
    max_rank: int = 50,
) -> list[HardNegativeExample]:
    """Create examples from the union of Task 1/2 rank-window candidates."""
    if min_rank < 1 or max_rank < min_rank:
        raise ValueError("hard-negative ranks must satisfy 1 <= min_rank <= max_rank")
    examples: list[HardNegativeExample] = []
    for record in train_claims.records:
        gold = {item.doc_id for item in record.evidence}
        runs = (bm25_run, dense_run) + ((scientific_run or {}),)
        candidates = list({
            doc_id
            for run in runs
            for rank, (doc_id, _) in enumerate(run.get(record.claim_id, []), 1)
            if min_rank <= rank <= max_rank and doc_id not in gold
        })
        for evidence in record.evidence:
            if candidates:
                negative_doc_id = candidates[0]
                examples.append(
                    HardNegativeExample(
                        record.claim_id,
                        record.claim,
                        evidence.doc_id,
                        document_text(corpus, evidence.doc_id),
                        negative_doc_id,
                        document_text(corpus, negative_doc_id),
                    )
                )
            else:
                raise ValueError(f"no hard negative for training claim {record.claim_id}")
    return examples


def batch_without_duplicates(
    pairs: Sequence[PositivePair], batch_size: int
) -> list[list[PositivePair]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches: list[list[PositivePair]] = []
    current: list[PositivePair] = []
    claims: set[int] = set()
    documents: set[int] = set()
    for pair in pairs:
        if pair.claim_id in claims or pair.doc_id in documents:
            if current:
                batches.append(current)
            current, claims, documents = [], set(), set()
        current.append(pair)
        claims.add(pair.claim_id)
        documents.add(pair.doc_id)
        if len(current) == batch_size:
            batches.append(current)
            current, claims, documents = [], set(), set()
    if current:
        batches.append(current)
    return batches


def build_hard_negative_pools(
    claims: Claims,
    runs: Mapping[str, Mapping[int, Sequence[tuple[int, float]]]],
    *,
    min_rank: int = 10,
    max_rank: int = 50,
) -> dict[int, list[int]]:
    """Pool Task 1/2 documents in the requested rank window, excluding gold."""
    if min_rank < 1 or max_rank < min_rank:
        raise ValueError("hard-negative ranks must satisfy 1 <= min_rank <= max_rank")
    pools: dict[int, list[int]] = {}
    for record in claims.records:
        gold = {item.doc_id for item in record.evidence}
        pool: set[int] = set()
        for run in runs.values():
            for rank, (doc_id, _) in enumerate(run.get(record.claim_id, []), 1):
                if min_rank <= rank <= max_rank and doc_id not in gold:
                    pool.add(doc_id)
        pools[record.claim_id] = sorted(pool)
    return pools


def sample_hard_negative_examples(
    positives: Sequence[PositivePair],
    claims: Claims,
    corpus: Corpus,
    pools: Mapping[int, Sequence[int]],
    *,
    seed: int,
) -> tuple[list[HardNegativeExample], int]:
    """Sample one negative per positive, falling back to a random non-gold doc."""
    gold_by_claim = {
        record.claim_id: {item.doc_id for item in record.evidence}
        for record in claims.records
    }
    all_doc_ids = [record.doc_id for record in corpus.records]
    rng = random.Random(seed)
    examples: list[HardNegativeExample] = []
    fallbacks = 0
    for pair in positives:
        candidates = list(pools.get(pair.claim_id, []))
        if not candidates:
            candidates = [
                doc_id for doc_id in all_doc_ids
                if doc_id not in gold_by_claim[pair.claim_id]
            ]
            fallbacks += 1
        negative_doc_id = rng.choice(candidates)
        examples.append(HardNegativeExample(
            pair.claim_id,
            pair.claim,
            pair.doc_id,
            pair.document,
            negative_doc_id,
            document_text(corpus, negative_doc_id),
        ))
    return examples, fallbacks


def sample_random_negative_examples(
    positives: Sequence[PositivePair],
    claims: Claims,
    corpus: Corpus,
    *,
    seed: int,
) -> list[HardNegativeExample]:
    """Sample one seeded non-gold document for every positive pair."""
    gold_by_claim = {
        record.claim_id: {item.doc_id for item in record.evidence}
        for record in claims.records
    }
    all_doc_ids = [record.doc_id for record in corpus.records]
    rng = random.Random(seed)
    examples: list[HardNegativeExample] = []
    for pair in positives:
        candidates = [
            doc_id for doc_id in all_doc_ids
            if doc_id not in gold_by_claim[pair.claim_id]
        ]
        if not candidates:
            raise ValueError(f"no random negative for training claim {pair.claim_id}")
        negative_doc_id = rng.choice(candidates)
        examples.append(HardNegativeExample(
            pair.claim_id,
            pair.claim,
            pair.doc_id,
            pair.document,
            negative_doc_id,
            document_text(corpus, negative_doc_id),
        ))
    return examples


def select_best_checkpoint(epoch_metrics: Mapping[int, float]) -> int:
    if not epoch_metrics:
        raise ValueError("epoch_metrics must not be empty")
    return max(epoch_metrics, key=lambda epoch: (float(epoch_metrics[epoch]), -epoch))


def _require_losses_api() -> Any:
    """Compatibility helper; training uses the local loss implementation below."""
    return importlib.import_module("sentence_transformers.losses")


def _load_training_run(
    path: Path, claims: Claims, corpus: Corpus
) -> dict[int, list[tuple[int, float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing training ranking artifact: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"training ranking artifact must be an object: {path}")
    expected = {record.claim_id for record in claims.records}
    doc_ids = {record.doc_id for record in corpus.records}
    output: dict[int, list[tuple[int, float]]] = {}
    for raw_claim_id, raw_ranking in raw.items():
        claim_id = int(raw_claim_id)
        if claim_id not in expected or not isinstance(raw_ranking, list):
            raise ValueError(f"invalid ranking for claim {claim_id}: {path}")
        seen: set[int] = set()
        ranking: list[tuple[int, float]] = []
        for item in raw_ranking:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"malformed ranking item for claim {claim_id}")
            doc_id, score = int(item[0]), float(item[1])
            if doc_id not in doc_ids or doc_id in seen:
                raise ValueError(f"invalid or duplicate document {doc_id}")
            seen.add(doc_id)
            ranking.append((doc_id, score))
        output[claim_id] = ranking
    if set(output) != expected:
        raise ValueError(f"training ranking artifact is missing claims: {path}")
    return output


class _Encoder:
    """Small transformer wrapper with separate query and document lengths."""

    def __init__(self, settings: FineTuneSettings) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        settings = FineTuneSettings(**{
            **settings.__dict__,
            "device": resolve_device(settings.device),
        })
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
        self.model = AutoModel.from_pretrained(settings.base_model).to(settings.device)
        parameter_count = sum(int(parameter.numel()) for parameter in self.model.parameters())
        if parameter_count > 400_000_000:
            raise ValueError(
                f"fine-tuning model has {parameter_count:,} parameters; "
                "the maximum is 400,000,000"
            )
        self.model.train()
        self.model.gradient_checkpointing_enable()
        self.settings = settings

    def encode(self, texts: Sequence[str], *, query: bool, grad: bool) -> Any:
        torch = self.torch
        max_length = (
            self.settings.max_query_length if query else self.settings.max_doc_length
        )
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.settings.device) for key, value in encoded.items()}
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            output = self.model(**encoded).last_hidden_state
            if self.settings.pooling == "cls":
                pooled = output[:, 0]
            elif self.settings.pooling == "mean":
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            else:
                raise ValueError("finetune.pooling must be 'cls' or 'mean'")
            return torch.nn.functional.normalize(pooled, p=2, dim=1)


def _query_text(claim: str, settings: FineTuneSettings) -> str:
    return f"{settings.query_prefix}{claim}"


def _doc_embeddings(
    encoder: _Encoder,
    corpus: Corpus,
    settings: FineTuneSettings,
) -> Any:
    """Encode the corpus in length-sorted batches (less padding) in original order."""
    import torch

    encoder.model.eval()
    documents = [document_text(corpus, record.doc_id) for record in corpus.records]
    order = sorted(range(len(documents)), key=lambda index: len(documents[index]))
    with torch.no_grad():
        chunks = [
            encoder.encode(
                [documents[index] for index in order[start : start + settings.batch_size]],
                query=False,
                grad=False,
            )
            for start in range(0, len(order), settings.batch_size)
        ]
        sorted_embeddings = torch.cat(chunks)
        restored = torch.empty_like(sorted_embeddings)
        restored[torch.tensor(order, device=sorted_embeddings.device)] = sorted_embeddings
    return restored


def _retrieval(
    encoder: _Encoder,
    corpus: Corpus,
    claims: Claims,
    settings: FineTuneSettings,
    doc_embeddings: Any | None = None,
) -> dict[int, list[tuple[int, float]]]:
    import torch

    encoder.model.eval()
    if doc_embeddings is None:
        doc_embeddings = _doc_embeddings(encoder, corpus, settings)
    doc_embeddings = doc_embeddings.to(settings.device)
    with torch.no_grad():
        output: dict[int, list[tuple[int, float]]] = {}
        for start in range(0, len(claims.records), settings.batch_size):
            records = claims.records[start : start + settings.batch_size]
            queries = [_query_text(record.claim, settings) for record in records]
            scores = encoder.encode(queries, query=True, grad=False) @ doc_embeddings.T
            for row, record in enumerate(records):
                order = torch.argsort(scores[row], descending=True, stable=True)[: settings.depth]
                output[record.claim_id] = [
                    (corpus.records[int(index)].doc_id, float(scores[row, index]))
                    for index in order
                ]
    encoder.model.train()
    return output


def _train(
    encoder: _Encoder,
    examples: Sequence[PositivePair] | Sequence[HardNegativeExample],
    settings: FineTuneSettings,
    corpus: Corpus,
    validation_claims: Claims,
    gold_by_claim: Mapping[int, set[int]],
) -> tuple[dict[int, dict[str, float]], dict[int, Any], dict[int, list[tuple[int, float]]], Any]:
    import torch

    if not examples:
        raise ValueError("fine-tuning requires training examples")
    optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=settings.learning_rate)
    steps = settings.epochs * ((len(examples) + settings.batch_size - 1) // settings.batch_size)
    warmup_steps = int(steps * settings.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01 if warmup_steps else 1.0,
        total_iters=max(1, warmup_steps),
    )
    curves: dict[int, dict[str, float]] = {}
    best_state: dict[int, Any] = {}
    best_run: dict[int, list[tuple[int, float]]] = {}
    best_embeddings: Any = None
    best_score = float("-inf")
    for epoch in range(1, settings.epochs + 1):
        encoder.model.train()
        shuffled = list(examples)
        random.Random(settings.seed + epoch).shuffle(shuffled)
        losses: list[float] = []
        for start in range(0, len(shuffled), settings.batch_size):
            batch = shuffled[start : start + settings.batch_size]
            queries = [_query_text(item.claim, settings) for item in batch]
            positives = [item.document if isinstance(item, PositivePair) else item.positive_document for item in batch]
            negatives = [
                "" if isinstance(item, PositivePair) else item.negative_document
                for item in batch
            ]
            q = encoder.encode(queries, query=True, grad=True)
            candidate_texts = positives + [text for text in negatives if text]
            candidates = encoder.encode(candidate_texts, query=False, grad=True)
            logits = (q @ candidates.T) * 20.0
            candidate_doc_ids = [
                item.doc_id if isinstance(item, PositivePair) else item.positive_doc_id
                for item in batch
            ] + [
                item.negative_doc_id for item in batch if not isinstance(item, PositivePair)
            ]
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for row, item in enumerate(batch):
                gold_docs = gold_by_claim[item.claim_id]
                for column, doc_id in enumerate(candidate_doc_ids):
                    if column != row and doc_id in gold_docs:
                        mask[row, column] = True
            logits = logits.masked_fill(mask, -1e4)
            targets = torch.arange(len(batch), device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if warmup_steps and optimizer.state[optimizer.param_groups[0]["params"][0]].get("step", 0) <= warmup_steps:
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
        embeddings = _doc_embeddings(encoder, corpus, settings)
        run = _retrieval(encoder, corpus, validation_claims, settings, doc_embeddings=embeddings)
        score = float(evaluate_retrieval(
            {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in run.items()},
            gold_documents(validation_claims),
        )["mean"]["ndcg@10"])
        curves[epoch] = {"train_loss": sum(losses) / len(losses), "val_ndcg@10": score}
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(encoder.model.state_dict())
            best_run = run
            best_embeddings = embeddings.detach().cpu()
    encoder.model.load_state_dict(best_state)
    return curves, best_state, best_run, best_embeddings


def _write_eval_run(
    encoder: _Encoder,
    corpus: Corpus,
    claims: Claims,
    settings: FineTuneSettings,
    path: Path,
    run_name: str,
    doc_embeddings: Any | None = None,
) -> dict[int, list[tuple[int, float]]]:
    run = _retrieval(encoder, corpus, claims, settings, doc_embeddings=doc_embeddings)
    write_trec(path, run, run_name, expected_claim_ids={record.claim_id for record in claims.records})
    return run


def run_task4(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    result_path = Path(config["paths"]["results_dir"]) / "task4_finetuning.json"
    curves_path = Path(config["paths"]["results_dir"]) / "task4_curves.json"
    seed = int(config["seed"])
    set_global_seed(seed)
    random.seed(seed)
    corpus = load_corpus(config["paths"]["corpus"])
    splits = load_all_claim_splits(config["paths"]["data_dir"])
    ft = config["finetune"]
    checkpoint_dir = project_path(config, ft["checkpoint_dir"])
    settings = FineTuneSettings(
        base_model=str(ft["base_model"]),
        output_dir=str(checkpoint_dir),
        learning_rate=float(ft["learning_rate"]),
        epochs=int(ft["epochs"]),
        batch_size=int(ft["batch_size"]),
        warmup_ratio=float(ft["warmup_ratio"]),
        max_doc_length=int(ft["max_doc_length"]),
        max_query_length=int(ft["max_query_length"]),
        device=resolve_device(str(ft["device"])),
        depth=int(ft["depth"]),
        hard_negative_min_rank=int(ft["hard_negative_min_rank"]),
        hard_negative_max_rank=int(ft["hard_negative_max_rank"]),
        seed=seed,
        query_prefix=str(config["dense"]["models"]["general"]["query_prefix"]),
        pooling=str(ft.get("pooling", "mean")),
    )
    positives = build_positive_pairs(splits["train"], corpus)
    if not positives:
        raise ValueError("Task 4 requires at least one positive training pair")
    cache_dir = Path(config["paths"]["cache_dir"])
    task1_result = json.loads(
        (Path(config["paths"]["results_dir"]) / "task1_bm25_tuning.json").read_text()
    )
    selected = task1_result["selected_settings"]
    bm25_settings = BM25Settings(
        k1=float(selected["k1"]),
        b=float(selected["b"]),
        stem=bool(selected["stem"]),
        stopwords=bool(selected["stopwords"]),
        fields=tuple(selected["fields"]),
        depth=100,
    )
    bm25_retriever = BM25Retriever(corpus, bm25_settings)
    bm25_train = bm25_retriever.retrieve(splits["train"], depth=100)
    dense_settings = _settings_from_config(config, "general")
    general_cache = cache_dir / f"emb_{_cache_name(dense_settings.name)}.npy"
    if not general_cache.is_file():
        raise FileNotFoundError(
            f"cached general dense corpus embeddings are required: {general_cache}"
        )
    dense_retriever = DenseRetriever(corpus, dense_settings)
    dense_train, _ = dense_retriever.retrieve(splits["train"], depth=100)
    hard_runs: dict[str, Mapping[int, Sequence[tuple[int, float]]]] = {
        "bm25": {claim_id: ranking[:50] for claim_id, ranking in bm25_train.items()},
        "dense_general": {claim_id: ranking[:50] for claim_id, ranking in dense_train.items()},
    }
    scientific_settings = _settings_from_config(config, "scientific")
    scientific_cache = cache_dir / f"emb_{_cache_name(scientific_settings.name)}.npy"
    scientific_retriever = None
    if scientific_cache.is_file():
        scientific_retriever = DenseRetriever(corpus, scientific_settings)
        scientific_train, _ = scientific_retriever.retrieve(splits["train"], depth=100)
        hard_runs["dense_scientific"] = {
            claim_id: ranking[:50] for claim_id, ranking in scientific_train.items()
        }
    del bm25_retriever, dense_retriever, scientific_retriever
    gc.collect()
    pools = build_hard_negative_pools(
        splits["train"],
        hard_runs,
        min_rank=settings.hard_negative_min_rank,
        max_rank=settings.hard_negative_max_rank,
    )
    pool_path = cache_dir / "hard_negs_train.json"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(json.dumps(pools, indent=2) + "\n", encoding="utf-8")
    hard, num_fallbacks = sample_hard_negative_examples(
        positives,
        splits["train"],
        corpus,
        pools,
        seed=seed,
    )
    import torch
    results_dir = Path(config["paths"]["results_dir"])
    prediction_dir = Path(config["paths"]["predictions_dir"])
    curves: dict[str, Any] = (
        json.loads(curves_path.read_text(encoding="utf-8"))
        if curves_path.is_file()
        else {}
    )
    gold_by_claim = {
        record.claim_id: {item.doc_id for item in record.evidence}
        for record in splits["train"].records
    }
    base_val_path = results_dir / "val_base.trec"
    if "base" not in curves or not base_val_path.is_file():
        base_encoder = _Encoder(settings)
        base_run = _retrieval(base_encoder, corpus, splits["val"], settings)
        curves["base"] = {
            "val_ndcg@10": float(evaluate_retrieval(
                {
                    claim_id: [doc_id for doc_id, _ in ranking]
                    for claim_id, ranking in base_run.items()
                },
                gold_documents(splits["val"]),
            )["mean"]["ndcg@10"])
        }
        write_trec(
            base_val_path,
            base_run,
            "base",
            expected_claim_ids={record.claim_id for record in splits["val"].records},
        )
    random_examples = sample_random_negative_examples(
        positives,
        splits["train"],
        corpus,
        seed=seed,
    )

    def train_run(
        name: str,
        examples: Sequence[HardNegativeExample],
        val_path: Path,
        eval_path: Path,
    ) -> None:
        if (
            name in curves
            and val_path.is_file()
            and eval_path.is_file()
            and "epochs_trained" in curves[name]
            and "best_epoch" in curves[name]
        ):
            return
        set_global_seed(seed)
        encoder = _Encoder(settings)
        run_curves, best_state, best_val, best_embeddings = _train(
            encoder,
            examples,
            settings,
            corpus,
            splits["val"],
            gold_by_claim,
        )
        best_epoch = select_best_checkpoint({
            epoch: values["val_ndcg@10"]
            for epoch, values in run_curves.items()
        })
        curves[name] = {
            "epochs_trained": settings.epochs,
            "best_epoch": best_epoch,
            **{str(epoch): values for epoch, values in run_curves.items()},
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_dir / f"{name}_best.pt")
        write_trec(
            val_path,
            best_val,
            name,
            expected_claim_ids={record.claim_id for record in splits["val"].records},
        )
        encoder.model.load_state_dict(best_state)
        _write_eval_run(
            encoder, corpus, splits["eval"], settings, eval_path, name,
            doc_embeddings=best_embeddings,
        )
        del encoder, best_state, best_embeddings
        gc.collect()
        if settings.device.startswith("cuda"):
            torch.cuda.empty_cache()

    train_run(
        "t4_random",
        random_examples,
        results_dir / "val_t4_random.trec",
        prediction_dir / "t4_random.trec",
    )
    train_run(
        "t4_hard",
        hard,
        results_dir / "val_t4_hard.trec",
        prediction_dir / "t4_hard.trec",
    )
    curves["mining_sources"] = sorted(hard_runs)
    curves["rank_range"] = [
        settings.hard_negative_min_rank,
        settings.hard_negative_max_rank,
    ]
    curves["mean_pool_size"] = sum(len(pool) for pool in pools.values()) / len(pools)
    curves["num_fallbacks"] = num_fallbacks
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path.write_text(json.dumps(curves, indent=2) + "\n", encoding="utf-8")
    result = {
        "base_model": settings.base_model,
        "training_examples": {
            "positive_pairs": len(positives),
            "random_negatives": len(random_examples),
            "hard_negatives": len(hard),
        },
        "base_validation_ndcg@10": curves["base"]["val_ndcg@10"],
        "selected_runs": {"random": "val_t4_random.trec", "hard": "val_t4_hard.trec"},
        "epochs": settings.epochs,
        "best_epochs": {
            name: curves[name]["best_epoch"]
            for name in ("t4_random", "t4_hard")
        },
        "mining_sources": sorted(hard_runs),
        "rank_range": [
            settings.hard_negative_min_rank,
            settings.hard_negative_max_rank,
        ],
        "mean_pool_size": curves["mean_pool_size"],
        "num_fallbacks": num_fallbacks,
    }
    result_path = results_dir / "task4_finetuning.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run_task4()
