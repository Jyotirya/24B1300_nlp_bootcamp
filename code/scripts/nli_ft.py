"""Train the optional SciFact selector and NLI models on training claims."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import DEFAULT_CONFIG, load_config, resolve_device, set_global_seed
from .data import Claims, Corpus, CorpusRecord, load_all_claim_splits, load_corpus
from .fuse import fuse_runs
from .verify import _select_premise_indices, _truncate_premise, read_label_mapping

LOGGER = logging.getLogger(__name__)

SentencePicker = Callable[[str, Sequence[str], int], list[str]]
PremiseBuilder = Callable[[str, CorpusRecord], str]


def _read_run(path: Path) -> dict[int, list[tuple[int, float]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(claim_id): [(int(doc_id), float(score)) for doc_id, score in ranking]
        for claim_id, ranking in raw.items()
    }


def _train_runs(config: Mapping[str, Any]) -> dict[int, list[tuple[int, float]]]:
    cache_dir = Path(config["paths"]["cache_dir"])
    bm25 = _read_run(cache_dir / "task1_bm25_train.json")
    dense = _read_run(cache_dir / "task2_general_train.json")
    return fuse_runs(bm25, dense, method="rrf", depth=10)


def _sentence_picker(config: Mapping[str, Any]) -> tuple[SentencePicker, PremiseBuilder]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model_config = config["dense"]["models"]["general"]
    model = SentenceTransformer(
        str(model_config["name"]),
        device=resolve_device(str(config["dense"]["device"])),
    )
    model.max_seq_length = int(config["dense"]["max_seq_length"])
    prefix = str(model_config.get("query_prefix", ""))
    sentence_count = int(config["verification"].get("premise_sentences", 3))
    max_tokens = int(config["verification"].get("premise_max_length", 384))
    tokenizer = getattr(model, "tokenizer", None)

    def pick(claim: str, sentences: Sequence[str], count: int = 2) -> list[str]:
        if not sentences:
            return []
        vectors = model.encode(
            [prefix + claim, *sentences],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = np.asarray(vectors[1:]) @ np.asarray(vectors[0])
        selected = _select_premise_indices(scores, count)
        return [sentences[index] for index in selected]

    def build_premise(claim: str, document: Any) -> str:
        sentences = list(document.abstract)
        if not sentences:
            return document.title
        text = " ".join(pick(claim, sentences, sentence_count))
        return _truncate_premise(text, tokenizer, max_tokens)

    return pick, build_premise


def _selector_premise_builder(
    selector: Any,
    config: Mapping[str, Any],
) -> PremiseBuilder:
    """Build NLI premises with the same trained selector used by Task 5."""
    verification_config = config["verification"]
    sentence_count = int(verification_config.get("premise_sentences", 3))
    max_tokens = int(verification_config.get("premise_max_length", 256))
    batch_size = int(verification_config.get("ft_batch_size", 16))
    tokenizer = getattr(selector, "tokenizer", None)
    selector.model.eval()

    def build_premise(claim: str, document: CorpusRecord) -> str:
        sentences = list(document.abstract)
        if not sentences:
            return document.title
        scores = selector.predict(
            [(claim, sentence) for sentence in sentences],
            batch_size=batch_size,
            show_progress_bar=False,
        )
        indices = _select_premise_indices(scores, sentence_count)
        text = " ".join(sentences[index] for index in indices)
        return _truncate_premise(text, tokenizer, max_tokens)

    return build_premise


def _training_examples(
    claims: Claims,
    corpus: Corpus,
    rankings: Mapping[int, Sequence[tuple[int, float]]],
    pick_sentences: SentencePicker,
    build_premise: PremiseBuilder,
    *,
    seed: int,
    nli_labels: Mapping[str, int],
    include_gold_rationales: bool = True,
    include_selector_examples: bool = True,
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    selector: list[tuple[str, str, int]] = []
    nli: list[tuple[str, str, int]] = []
    for record in claims.records:
        evidence = [
            item for item in record.evidence
            if item.label in {"SUPPORT", "CONTRADICT"}
        ]
        if not evidence:
            for doc_id, _ in rankings.get(record.claim_id, [])[:2]:
                doc_index = corpus.id_to_index.get(doc_id)
                if doc_index is None:
                    continue
                premise = build_premise(record.claim, corpus.records[doc_index])
                if premise:
                    nli.append((premise, record.claim, nli_labels["neutral"]))
            continue
        gold_docs = {item.doc_id for item in evidence}
        non_gold = [
            doc_id for doc_id, _ in rankings.get(record.claim_id, [])[:10]
            if doc_id not in gold_docs
        ]
        rng = random.Random(seed + record.claim_id)
        for item in evidence:
            corpus_index = corpus.id_to_index[item.doc_id]
            gold_sentences = corpus.records[corpus_index].abstract
            rationale = [
                gold_sentences[index] for index in item.sentences
                if 0 <= index < len(gold_sentences)
            ]
            if not rationale:
                continue
            if include_selector_examples:
                other = [
                    sentence for index, sentence in enumerate(gold_sentences)
                    if index not in set(item.sentences)
                ]
                sampled_other = rng.sample(other, min(2, len(other))) if other else []
                selector.extend(
                    (record.claim, sentence, 1) for sentence in rationale
                )
                selector.extend(
                    (record.claim, sentence, 0)
                    for sentence in sampled_other
                )
            negative_index = corpus.id_to_index[non_gold[0]] if non_gold else None
            if include_selector_examples and negative_index is not None:
                negative_sentences = corpus.records[negative_index].abstract
                selector.extend(
                    (record.claim, sentence, 0)
                    for sentence in pick_sentences(
                        record.claim, negative_sentences, count=2
                    )
                )
            premise = build_premise(record.claim, corpus.records[corpus_index])
            nli.append((
                premise,
                record.claim,
                nli_labels[
                    "entailment" if item.label == "SUPPORT" else "contradiction"
                ],
            ))
            if include_gold_rationales:
                nli.append((
                    " ".join(rationale),
                    record.claim,
                    nli_labels[
                        "entailment" if item.label == "SUPPORT" else "contradiction"
                    ],
                ))
            negative_premise = (
                build_premise(record.claim, corpus.records[negative_index])
                if negative_index is not None else ""
            )
            if negative_premise:
                nli.append((
                    negative_premise, record.claim, nli_labels["neutral"]
                ))
    return selector, nli


def _fit(
    model: Any,
    tokenizer: Any,
    examples: Sequence[tuple[str, str, int]],
    *,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    nli: bool,
    seed: int,
) -> list[float]:
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps = epochs * max(1, (len(examples) + batch_size - 1) // batch_size)
    warmup_steps = max(1, int(steps * warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and not nli)
    losses: list[float] = []
    for epoch in range(epochs):
        order = list(range(len(examples)))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), batch_size):
            batch = [examples[index] for index in order[start:start + batch_size]]
            first = [item[0] for item in batch]
            second = [item[1] for item in batch]
            tokens = tokenizer(
                first, second, padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            labels = torch.tensor(
                [item[2] for item in batch],
                dtype=torch.long if nli else torch.float32,
                device=device,
            )
            optimizer.zero_grad()
            context = (
                torch.autocast(
                    device_type="cuda", dtype=torch.float16,
                    enabled=device.startswith("cuda"),
                )
                if not nli else nullcontext()
            )
            with context:
                logits = model(**tokens).logits
                if nli:
                    loss = torch.nn.functional.cross_entropy(
                        logits, labels,
                    )
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.reshape(-1), labels,
                    )
            if not torch.isfinite(loss):
                LOGGER.warning("skipping NaN or infinite loss at epoch %d", epoch + 1)
                continue
            optimizer.zero_grad()
            if nli:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    LOGGER.warning("skipping non-finite gradient at epoch %d", epoch + 1)
                    optimizer.zero_grad()
                    continue
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    LOGGER.warning("skipping non-finite gradient at epoch %d", epoch + 1)
                    optimizer.zero_grad()
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
            if optimizer.state[optimizer.param_groups[0]["params"][0]].get("step", 0) <= warmup_steps:
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
    return losses


def _run(config_path: str | Path) -> dict[str, Any]:
    import torch
    from sentence_transformers import CrossEncoder

    config = load_config(config_path)
    set_global_seed(int(config["seed"]))
    corpus = load_corpus(config["paths"]["corpus"])
    splits = load_all_claim_splits(config["paths"]["data_dir"])
    rankings = _train_runs(config)
    picker, premise_builder = _sentence_picker(config)
    ft = config["verification"]["ft"]
    mix_gold_rationales = bool(ft.get("nli_gold_rationale_mix", True))
    device = resolve_device(str(config["verification"]["device"]))
    selector = CrossEncoder(
        str(ft["selector_model_name"]), device=device, num_labels=1
    )
    nli = CrossEncoder(
        str(config["verification"]["model_name"]), device=device, num_labels=3
    )
    label_indices = read_label_mapping(nli.model.config)
    selector_examples, nli_examples = _training_examples(
        splits["train"], corpus, rankings, picker, premise_builder,
        seed=int(config["seed"]), nli_labels=label_indices,
        include_gold_rationales=mix_gold_rationales,
    )
    nli_label_counts = Counter(label for _, _, label in nli_examples)
    missing_labels = [
        name for name, index in label_indices.items()
        if nli_label_counts[index] == 0
    ]
    if missing_labels:
        raise ValueError(
            "NLI fine-tuning has no training examples for "
            f"{', '.join(missing_labels)}; check training retrieval runs and evidence."
        )
    selector_losses = _fit(
        selector.model, selector.tokenizer, selector_examples,
        device=device, epochs=2, batch_size=32, learning_rate=3e-5,
        warmup_ratio=0.0, nli=False,
        seed=int(config["seed"]),
    )
    selector_premise_builder = _selector_premise_builder(selector, config)
    _, selector_nli_examples = _training_examples(
        splits["train"], corpus, rankings, picker, selector_premise_builder,
        seed=int(config["seed"]), nli_labels=label_indices,
        include_gold_rationales=False,
        include_selector_examples=False,
    )
    nli_examples.extend(selector_nli_examples)
    nli_losses = _fit(
        nli.model, nli.tokenizer, nli_examples,
        device=device, epochs=3, batch_size=16, learning_rate=1e-5,
        warmup_ratio=0.1, nli=True,
        seed=int(config["seed"]),
    )
    output_dir = Path(config["paths"]["cache_dir"]) / "nli_ft"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(selector.model.state_dict(), output_dir / "selector.pt")
    torch.save(nli.model.state_dict(), output_dir / "nli.pt")
    result = {
        "selector_examples": len(selector_examples),
        "nli_examples": len(nli_examples),
        "nli_label_counts": {
            name: nli_label_counts[index]
            for name, index in label_indices.items()
        },
        "nli_gold_rationale_mix": mix_gold_rationales,
        "nli_training_premises": [
            "dense_sentence_selector",
            "fine_tuned_selector",
        ],
        "model_name": str(config["verification"]["model_name"]),
        "selector_loss": selector_losses,
        "nli_loss": nli_losses,
        "label_indices": label_indices,
    }
    result_path = Path(config["paths"]["results_dir"]) / "nli_ft.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    _run(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
