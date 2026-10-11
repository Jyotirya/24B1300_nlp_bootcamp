"""Optional fine-tuning of the Task 3 cross-encoder reranker."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import DEFAULT_CONFIG, load_config, resolve_device, set_global_seed
from .data import Claims, Corpus, load_all_claim_splits, load_corpus
from .dense import DenseRetriever, _settings_from_config
from .fuse import fuse_runs
from .metrics import evaluate_retrieval, gold_documents
from .rerank import (
    RerankSettings,
    count_cross_encoder_parameters,
    document_text,
)
from .retrieve import BM25Retriever, BM25Settings


def _load_first_stage(config: Mapping[str, Any], corpus: Corpus, splits: Mapping[str, Claims]) -> dict[str, dict[int, list[tuple[int, float]]]]:
    result = json.loads(
        (Path(config["paths"]["results_dir"]) / "task1_bm25_tuning.json").read_text()
    )
    settings = result["selected_settings"]
    bm25 = BM25Retriever(corpus, BM25Settings(
        k1=float(settings["k1"]),
        b=float(settings["b"]),
        stem=bool(settings["stem"]),
        stopwords=bool(settings["stopwords"]),
        fields=tuple(settings["fields"]),
        depth=200,
    ))
    dense = DenseRetriever(corpus, _settings_from_config(config, "general"))
    runs: dict[str, dict[int, list[tuple[int, float]]]] = {}
    for split in ("train", "val", "eval"):
        bm25_run = bm25.retrieve(splits[split], depth=200)
        dense_run, _ = dense.retrieve(splits[split], depth=200)
        runs[split] = fuse_runs(bm25_run, dense_run, method="rrf", depth=200)
    return runs


def _examples(
    claims: Claims,
    corpus: Corpus,
    first_stage: Mapping[int, Sequence[tuple[int, float]]],
    *,
    seed: int,
) -> list[tuple[str, str, int]]:
    examples: list[tuple[str, str, int]] = []
    for record in claims.records:
        gold = {item.doc_id for item in record.evidence}
        if not gold:
            continue
        negatives = [doc_id for doc_id, _ in first_stage.get(record.claim_id, [])[:30] if doc_id not in gold]
        if len(negatives) < 7:
            raise ValueError(
                f"claim {record.claim_id} has fewer than 7 eligible negatives in top 30"
            )
        rng = random.Random(seed + record.claim_id)
        for item in record.evidence:
            examples.append((record.claim, document_text(corpus, item.doc_id), 1))
            for doc_id in rng.sample(negatives, 7):
                examples.append((record.claim, document_text(corpus, doc_id), 0))
    return examples


def _score(
    model: Any,
    tokenizer: Any,
    device: str,
    claims: Claims,
    corpus: Corpus,
    rankings: Mapping[int, Sequence[tuple[int, float]]],
    *,
    max_length: int,
    batch_size: int,
    cache_path: Path,
    reuse_cache: bool = True,
) -> dict[str, float]:
    import torch

    if reuse_cache and cache_path.is_file():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return {str(key): float(value) for key, value in raw.items()}
    scores: dict[str, float] = {}
    pending: list[tuple[str, str, str]] = []
    for record in claims.records:
        for doc_id, _ in rankings.get(record.claim_id, [])[:50]:
            pending.append((
                f"{record.claim_id}:{doc_id}",
                record.claim,
                document_text(corpus, doc_id),
            ))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            tokens = tokenizer(
                [item[1] for item in batch],
                [item[2] for item in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            logits = model(**tokens).logits.reshape(-1)
            scores.update({
                item[0]: float(value)
                for item, value in zip(batch, logits.detach().cpu())
            })
            if len(scores) % 50 == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores


def _cached_rankings(path: Path, claims: Claims) -> dict[int, list[tuple[int, float]]]:
    if not path.is_file():
        raise FileNotFoundError("run Task 3 first")
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_claim_ids = {record.claim_id for record in claims.records}
    rankings: dict[int, list[tuple[int, float]]] = {
        claim_id: [] for claim_id in expected_claim_ids
    }
    for record in claims.records:
        prefix = f"{record.claim_id}:"
        for key, value in raw.items():
            key_text = str(key)
            if key_text.startswith(prefix):
                parts = key_text.split(":")
                if len(parts) != 2:
                    raise ValueError(f"malformed CE cache key: {key_text}")
                rankings[record.claim_id].append((int(parts[1]), float(value)))
    if not all(rankings.values()):
        raise ValueError("Task 3 CE caches are missing claim-document keys")
    return rankings


def _run(config_path: str | Path) -> dict[str, Any]:
    import torch
    from sentence_transformers import CrossEncoder

    config = load_config(config_path)
    set_global_seed(int(config["seed"]))
    corpus = load_corpus(config["paths"]["corpus"])
    splits = load_all_claim_splits(config["paths"]["data_dir"])
    device = resolve_device(str(config["rerank"]["device"]))
    settings = RerankSettings(
        name=str(config["rerank"]["model_name"]),
        batch_size=int(config["rerank"]["finetune"]["batch_size"]),
        device=device,
        max_k=int(config["rerank"]["max_k"]),
        max_length=int(config["rerank"]["finetune"]["max_length"]),
    )
    backend = CrossEncoder(settings.name, device=device, num_labels=1)
    assert_size = count_cross_encoder_parameters(backend)
    if assert_size > 400_000_000:
        raise ValueError("fine-tuned reranker exceeds the 400M parameter limit")
    model = backend.model
    tokenizer = backend.tokenizer
    first_stage = _load_first_stage(config, corpus, splits)
    train_source = _examples(
        splits["train"], corpus, first_stage["train"], seed=int(config["seed"])
    )
    ft = config["rerank"]["finetune"]
    batch_size = int(ft["batch_size"])
    epochs = int(ft["epochs"])
    max_length = int(ft["max_length"])
    total_steps = epochs * ((len(train_source) + batch_size - 1) // batch_size)
    warmup_steps = int(total_steps * float(ft["warmup_ratio"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(ft["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            step / max(1, warmup_steps)
            if step < warmup_steps
            else max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    curves: dict[str, Any] = {}
    best_score = float("-inf")
    best_state: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        examples = _examples(
            splits["train"], corpus, first_stage["train"],
            seed=int(config["seed"]) + epoch,
        )
        random.Random(int(config["seed"]) + epoch).shuffle(examples)
        losses: list[float] = []
        model.train()
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            tokens = tokenizer(
                [item[0] for item in batch],
                [item[1] for item in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            labels = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.startswith("cuda"),
            ):
                logits = model(**tokens).logits.reshape(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        zero_val_path = Path(config["paths"]["cache_dir"]) / "ce_val.json"
        zero_eval_path = Path(config["paths"]["cache_dir"]) / "ce_eval.json"
        val_rankings = _cached_rankings(zero_val_path, splits["val"])
        eval_rankings = _cached_rankings(zero_eval_path, splits["eval"])
        cache_path = Path(config["paths"]["cache_dir"]) / "ce_ft" / "ce_val.json"
        val_scores = _score(
            model, tokenizer, device, splits["val"], corpus, val_rankings,
            max_length=max_length, batch_size=batch_size, cache_path=cache_path,
            reuse_cache=False,
        )
        rankings = {
            record.claim_id: sorted(
                (
                    (doc_id, val_scores[f"{record.claim_id}:{doc_id}"])
                    for doc_id, _ in val_rankings[record.claim_id]
                ),
                key=lambda item: (-item[1], item[0]),
            )
            for record in splits["val"].records
        }
        score = float(evaluate_retrieval(
            {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in rankings.items()},
            gold_documents(splits["val"]),
        )["mean"]["ndcg@10"])
        curves[str(epoch)] = {
            "train_loss": sum(losses) / len(losses),
            "val_ndcg@10": score,
        }
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise ValueError("fine-tuning produced no checkpoint")
    checkpoint = Path(config["paths"]["cache_dir"]) / "ce_ft"
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint / "model.pt")
    model.load_state_dict(best_state)
    _score(
        model, tokenizer, device, splits["val"], corpus, val_rankings,
        max_length=max_length, batch_size=batch_size,
        cache_path=checkpoint / "ce_val.json",
        reuse_cache=False,
    )
    eval_cache = checkpoint / "ce_eval.json"
    _score(
        model, tokenizer, device, splits["eval"], corpus, eval_rankings,
        max_length=max_length, batch_size=batch_size, cache_path=eval_cache,
        reuse_cache=False,
    )
    results_path = Path(config["paths"]["results_dir"]) / "rerank_ft_curves.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(curves, indent=2) + "\n", encoding="utf-8")
    return curves


def main() -> None:
    _run(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
