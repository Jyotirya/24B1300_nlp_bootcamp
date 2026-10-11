"""NLI claim verification and validation threshold selection."""

from __future__ import annotations

import csv
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from .common import DEFAULT_CONFIG, load_config, project_path, resolve_device, set_global_seed
from .metrics import read_trec
from .data import Claims, Corpus, load_all_claim_splits, load_corpus
from .dense import MAX_MODEL_PARAMETERS
from .errors import ALLOWED_CATEGORIES, classify_error

Verification = tuple[int, int, str]
Ranking = Sequence[tuple[int, float]]


@dataclass(frozen=True)
class VerificationSettings:
    model_name: str = "cross-encoder/nli-deberta-v3-small"
    batch_size: int = 32
    device: str = "auto"
    candidate_depth: int = 100
    support_threshold: float = 0.5
    contradict_threshold: float = 0.5
    gate: float = 0.5


def read_label_mapping(config: Any) -> dict[str, int]:
    """Map model id2label values to entailment, contradiction, and neutral."""
    id2label = getattr(config, "id2label", None)
    if not isinstance(id2label, Mapping):
        raise ValueError("NLI model config must expose an id2label mapping")
    mapping: dict[str, int] = {}
    for index, label in id2label.items():
        normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
        if "entail" in normalized or "support" in normalized:
            key = "entailment"
        elif "contradict" in normalized:
            key = "contradiction"
        elif "neutral" in normalized:
            key = "neutral"
        else:
            raise ValueError(f"uninterpretable NLI label: {label!r}")
        if key in mapping:
            raise ValueError(f"duplicate NLI class mapping for {key}")
        mapping[key] = int(index)
    if set(mapping) != {"entailment", "contradiction", "neutral"}:
        raise ValueError("NLI id2label must identify entailment, contradiction, and neutral")
    return mapping


def count_nli_parameters(model: Any) -> int:
    """Count actual NLI model parameters and enforce the 400M limit."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        inner = getattr(model, "model", None)
        parameters = getattr(inner, "parameters", None)
    if not callable(parameters):
        raise TypeError("NLI model does not expose parameters()")
    count = sum(int(parameter.numel()) for parameter in parameters())
    if count > MAX_MODEL_PARAMETERS:
        raise ValueError(
            f"NLI model has {count:,} parameters, exceeding "
            f"{MAX_MODEL_PARAMETERS:,}"
        )
    return count


def verification_metrics(
    predictions: Collection[Verification],
    gold: Collection[Verification],
) -> dict[str, float]:
    """Calculate micro precision, recall, and F1 over verification triples."""
    predicted_set, gold_set = set(predictions), set(gold)
    true_positive = len(predicted_set & gold_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compare_nli_validation_f1(
    variant_scores: Mapping[str, float],
) -> dict[str, float] | None:
    """Compare fine-tuned NLI with the zero-shot whole-abstract baseline."""
    baseline = variant_scores.get("zero_shot_whole_abstract")
    fine_tuned = variant_scores.get("fine_tuned_nli")
    if baseline is None or fine_tuned is None:
        return None
    return {
        "whole_abstract_baseline_f1": float(baseline),
        "fine_tuned_nli_f1": float(fine_tuned),
        "f1_delta": float(fine_tuned) - float(baseline),
    }


def limit_predictions(predictions: Sequence[Verification], maximum: int = 3) -> list[Verification]:
    """Keep at most ``maximum`` predictions per claim in stable input order."""
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    counts: dict[int, int] = {}
    limited: list[Verification] = []
    for prediction in predictions:
        claim_id = prediction[0]
        if counts.get(claim_id, 0) < maximum:
            limited.append(prediction)
            counts[claim_id] = counts.get(claim_id, 0) + 1
    return limited


def whole_abstract_premise(claim: str, document: Any) -> str:
    """Fallback premise builder: use the full title + abstract text."""
    return f"{document.title}\n{' '.join(document.abstract)}"


def _best_verification_label(
    doc_id: int,
    entailment: float,
    contradiction: float,
    support_threshold: float,
    contradict_threshold: float,
) -> tuple[int, str, float] | None:
    """Choose at most one threshold-qualified NLI label for a document."""
    candidates: list[tuple[int, str, float]] = []
    if entailment >= support_threshold:
        candidates.append((doc_id, "SUPPORT", entailment))
    if contradiction >= contradict_threshold:
        candidates.append((doc_id, "CONTRADICT", contradiction))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (item[2], item[1] == "SUPPORT"),
    )


def _select_premise_indices(scores: Sequence[float], count: int) -> list[int]:
    """Select highest-scoring sentences, then restore their document order."""
    ranked = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), index),
    )[:count]
    return sorted(ranked)


def _truncate_premise(text: str, tokenizer: Any, max_tokens: int) -> str:
    if tokenizer is not None:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_tokens,
        )
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    return " ".join(text.split()[:max_tokens])


def select_thresholds(
    settings_grid: Sequence[tuple[float, ...]],
    evaluator: Any,
    *,
    prefer_coverage_on_ties: bool = False,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Select settings by validation F1, optionally preferring coverage on ties."""
    if not settings_grid:
        raise ValueError("threshold grid must not be empty")
    results: dict[str, Any] = {}
    for setting in settings_grid:
        metrics = evaluator(*setting)
        results[":".join(map(str, setting))] = metrics

    def selection_key(setting: tuple[float, ...]) -> tuple[float, ...]:
        metrics = results[":".join(map(str, setting))]
        tie_break = (
            (
                float(metrics.get("recall", 0.0)),
                float(metrics.get("prediction_count", 0.0)),
            )
            if prefer_coverage_on_ties
            else ()
        )
        return (
            float(metrics["f1"]),
            *tie_break,
            *setting[:-1],
            -setting[-1],
        )

    best = max(
        settings_grid,
        key=selection_key,
    )
    return best, results


def verification_threshold_grid(
    verification_config: Mapping[str, Any],
) -> list[tuple[float, float, float, int]]:
    """Build verification candidates from the configured threshold/depth lists."""
    support_values = tuple(
        float(value) for value in verification_config["support_thresholds"]
    )
    contradict_values = tuple(
        float(value) for value in verification_config["contradict_thresholds"]
    )
    gate_values = tuple(
        float(value)
        for value in verification_config.get(
            "gate_thresholds", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        )
    )
    depth_values = tuple(
        int(value) for value in verification_config["candidate_depths"]
    )
    if not support_values or not contradict_values or not gate_values or not depth_values:
        raise ValueError(
            "verification support_thresholds, contradict_thresholds, and "
            "gate_thresholds, and candidate_depths must not be empty"
        )
    return [
        (support, contradict, gate, depth)
        for support in support_values
        for contradict in contradict_values
        for gate in gate_values
        for depth in depth_values
    ]


def write_verification_csv(path: str | Path, predictions: Collection[Verification]) -> None:
    """Write exact checker-compatible verification CSV rows."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["claim_id", "doc_id", "label"])
        for claim_id, doc_id, label in sorted(predictions):
            if label not in {"SUPPORT", "CONTRADICT"}:
                raise ValueError(f"invalid verification label: {label}")
            writer.writerow([claim_id, doc_id, label])


def write_error_csv(path: str | Path, errors: Mapping[int, str]) -> None:
    """Write only incorrect validation claims using the required header."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["claim_id", "category"])
        for claim_id, category in sorted(errors.items()):
            if category not in ALLOWED_CATEGORIES:
                raise ValueError(f"invalid error category: {category}")
            writer.writerow([claim_id, category])


class NLIBackend:
    """Lazy CrossEncoder NLI backend with model-label introspection."""

    def __init__(self, settings: VerificationSettings, checkpoint: str | Path | None = None) -> None:
        settings = VerificationSettings(**{
            **settings.__dict__,
            "device": resolve_device(settings.device),
        })
        self.cache_model_identity: dict[str, Any] = {"model_name": settings.model_name}
        sentence_transformers = importlib.import_module("sentence_transformers")
        self.model = sentence_transformers.CrossEncoder(
            settings.model_name,
            device=settings.device,
            num_labels=3,
        )
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"NLI checkpoint not found: {checkpoint_path}")
            checkpoint_stat = checkpoint_path.stat()
            self.cache_model_identity["checkpoint"] = {
                "path": str(checkpoint_path.resolve()),
                "size": checkpoint_stat.st_size,
                "mtime_ns": checkpoint_stat.st_mtime_ns,
            }
            import torch
            self.model.model.load_state_dict(torch.load(
                checkpoint_path,
                map_location=settings.device,
                weights_only=True,
            ))
        self.settings = settings
        self.parameter_count = count_nli_parameters(self.model)
        config = getattr(getattr(self.model, "model", None), "config", None)
        self.label_indices = read_label_mapping(config)

    def predict(
        self,
        claims: Claims,
        rankings: Mapping[int, Ranking],
        corpus: Corpus,
        premise_builder: Any | None = None,
    ) -> tuple[list[Verification], float]:
        """Run NLI on the configured candidate depth, with a premise builder for sentence selection."""
        if premise_builder is None:
            premise_builder = whole_abstract_premise
        pairs: list[tuple[int, int, str, str]] = []
        for record in claims.records:
            for doc_id, _ in rankings.get(record.claim_id, [])[: self.settings.candidate_depth]:
                index = corpus.id_to_index.get(doc_id)
                if index is None:
                    raise ValueError(f"ranking contains unknown doc_id {doc_id}")
                document = corpus.records[index]
                pairs.append((
                    record.claim_id,
                    doc_id,
                    premise_builder(record.claim, document),
                    record.claim,
                ))
        if not pairs:
            return [], 0.0
        start = time.perf_counter()
        probabilities = self.model.predict(
            [(premise, hypothesis) for _, _, premise, hypothesis in pairs],
            batch_size=self.settings.batch_size,
            apply_softmax=True,
            show_progress_bar=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        predictions: list[Verification] = []
        offset = 0
        for record in claims.records:
            accepted_by_doc: dict[int, tuple[int, str, float]] = {}
            count = len(rankings.get(record.claim_id, [])[: self.settings.candidate_depth])
            for _, doc_id, _, _ in pairs[offset : offset + count]:
                scores = probabilities[offset]
                offset += 1
                entailment = float(scores[self.label_indices["entailment"]])
                contradiction = float(scores[self.label_indices["contradiction"]])
                selected = _best_verification_label(
                    doc_id,
                    entailment,
                    contradiction,
                    self.settings.support_threshold,
                    self.settings.contradict_threshold,
                )
                if selected is not None and (
                    doc_id not in accepted_by_doc
                    or selected[2] > accepted_by_doc[doc_id][2]
                ):
                    accepted_by_doc[doc_id] = selected
            accepted = list(accepted_by_doc.values())
            accepted.sort(key=lambda item: (-item[2], item[0], item[1]))
            predictions.extend(limit_predictions(
                [(record.claim_id, doc_id, label) for doc_id, label, _ in accepted],
                maximum=3,
            ))
        return predictions, elapsed_ms

    def score_cache(
        self,
        claims: Claims,
        rankings: Mapping[int, Ranking],
        corpus: Corpus,
        cache_path: str | Path,
        premise_builder: Any,
        *,
        cache_key: str,
    ) -> tuple[dict[int, list[dict[str, Any]]], float]:
        """Refresh probabilities for the configured candidate-depth window, not just the top 5."""
        path = Path(cache_path)
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        expected_metadata = {
            "model": self.cache_model_identity,
            "premise": cache_key,
        }
        existing: dict[int, dict[int, dict[str, Any]]] = {}
        if path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata == expected_metadata:
                raw = json.loads(path.read_text(encoding="utf-8"))
                for raw_claim_id, raw_entries in raw.items():
                    claim_id = int(raw_claim_id)
                    existing[claim_id] = {
                        int(entry["doc_id"]): dict(entry)
                        for entry in raw_entries
                    }
        depth = max(1, self.settings.candidate_depth)
        pairs: list[tuple[int, int, str, str]] = []
        for record in claims.records:
            for doc_id, _ in rankings.get(record.claim_id, [])[:depth]:
                if doc_id in existing.get(record.claim_id, {}):
                    continue
                index = corpus.id_to_index.get(doc_id)
                if index is None:
                    raise ValueError(f"ranking contains unknown doc_id {doc_id}")
                document = corpus.records[index]
                premise = premise_builder(record.claim, document)
                pairs.append((record.claim_id, doc_id, premise, record.claim))
        probabilities: Sequence[Sequence[float]] = []
        start = time.perf_counter()
        if pairs:
            probabilities = self.model.predict(
                [(premise, hypothesis) for _, _, premise, hypothesis in pairs],
                batch_size=self.settings.batch_size,
                apply_softmax=True,
                show_progress_bar=False,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        refreshed: dict[int, list[dict[str, Any]]] = {}
        pair_scores = {
            (claim_id, doc_id): [float(value) for value in probabilities[index]]
            for index, (claim_id, doc_id, _, _) in enumerate(pairs)
        }
        for record in claims.records:
            entries: list[dict[str, Any]] = []
            for doc_id, _ in rankings.get(record.claim_id, [])[: max(1, self.settings.candidate_depth)]:
                entry = existing.get(record.claim_id, {}).get(doc_id)
                if entry is None:
                    entry = {
                        "doc_id": doc_id,
                        "probabilities": pair_scores[(record.claim_id, doc_id)],
                    }
                entries.append(entry)
            refreshed[record.claim_id] = entries
            if len(refreshed) % 50 == 0:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(refreshed, indent=2) + "\n", encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(refreshed, indent=2) + "\n", encoding="utf-8")
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        return refreshed, elapsed_ms

    def from_cache(
        self,
        claims: Claims,
        cached: Mapping[int, Sequence[Mapping[str, Any]]],
    ) -> list[Verification]:
        predictions: list[Verification] = []
        for record in claims.records:
            entries = list(cached.get(record.claim_id, []))[: self.settings.candidate_depth]
            best_confidence = max(
                (max(
                    float(entry["probabilities"][self.label_indices["entailment"]]),
                    float(entry["probabilities"][self.label_indices["contradiction"]]),
                ) for entry in entries),
                default=0.0,
            )
            if best_confidence < self.settings.gate:
                continue
            accepted_by_doc: dict[int, tuple[int, str, float]] = {}
            for entry in entries:
                scores = entry["probabilities"]
                entailment = float(scores[self.label_indices["entailment"]])
                contradiction = float(scores[self.label_indices["contradiction"]])
                doc_id = int(entry["doc_id"])
                selected = _best_verification_label(
                    doc_id,
                    entailment,
                    contradiction,
                    self.settings.support_threshold,
                    self.settings.contradict_threshold,
                )
                if selected is not None and (
                    doc_id not in accepted_by_doc
                    or selected[2] > accepted_by_doc[doc_id][2]
                ):
                    accepted_by_doc[doc_id] = selected
            accepted = list(accepted_by_doc.values())
            accepted.sort(key=lambda item: (-item[2], item[0], item[1]))
            predictions.extend(limit_predictions(
                [(record.claim_id, doc_id, label) for doc_id, label, _ in accepted],
                maximum=3,
            ))
        return predictions


def _build_premise_selector(config: Mapping[str, Any]) -> Any:
    """Build a general-dense cosine selector for NLI premises."""
    sentence_transformers = importlib.import_module("sentence_transformers")
    model_config = config["dense"]["models"]["general"]
    model = sentence_transformers.SentenceTransformer(
        str(model_config["name"]),
        device=resolve_device(str(config["dense"]["device"])),
    )
    model.max_seq_length = int(config["dense"]["max_seq_length"])
    prefix = str(model_config.get("query_prefix", ""))
    sentence_count = int(config["verification"].get("premise_sentences", 3))
    max_tokens = int(config["verification"].get("premise_max_length", 384))

    def select(claim: str, document: Any) -> str:
        sentences = list(document.abstract)
        if not sentences:
            return document.title
        embeddings = model.encode(
            [prefix + claim, *sentences],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = embeddings[1:] @ embeddings[0]
        selected = _select_premise_indices(scores, sentence_count)
        text = " ".join(sentences[index] for index in selected)
        return _truncate_premise(text, getattr(model, "tokenizer", None), max_tokens)

    return select


def _build_selector_premise_builder(
    config: Mapping[str, Any],
    checkpoint: str | Path,
) -> Any:
    """Build a trained sentence selector that returns top sentences in order."""
    sentence_transformers = importlib.import_module("sentence_transformers")
    import torch

    ft = config["verification"]["ft"]
    model = sentence_transformers.CrossEncoder(
        str(ft["selector_model_name"]),
        device=resolve_device(str(config["verification"]["device"])),
        num_labels=1,
    )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"selector checkpoint not found: {checkpoint_path}")
    model.model.load_state_dict(torch.load(
        checkpoint_path,
        map_location=resolve_device(str(config["verification"]["device"])),
        weights_only=True,
    ))
    model.model.eval()
    sentence_count = int(config["verification"].get("premise_sentences", 3))
    max_tokens = int(config["verification"].get("premise_max_length", 256))
    tokenizer = getattr(model, "tokenizer", None)

    def select(claim: str, document: Any) -> str:
        sentences = list(document.abstract)
        if not sentences:
            return document.title
        scores = model.predict(
            [(claim, sentence) for sentence in sentences],
            batch_size=int(config["verification"].get("ft_batch_size", 16)),
            show_progress_bar=False,
        )
        chosen = sorted(
            sorted(
                range(len(sentences)),
                key=lambda index: (-float(scores[index]), index),
            )[:sentence_count]
        )
        text = " ".join(sentences[index] for index in chosen)
        if tokenizer is not None:
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_tokens,
            )
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        return text

    def gold_sentence_rate(claims: Claims, corpus: Corpus) -> float:
        total = 0
        hits = 0
        for record in claims.records:
            for evidence in record.evidence:
                index = corpus.id_to_index.get(evidence.doc_id)
                if index is None:
                    continue
                sentences = corpus.records[index].abstract
                if not sentences:
                    continue
                scores = model.predict(
                    [(record.claim, sentence) for sentence in sentences],
                    batch_size=int(config["verification"].get("ft_batch_size", 16)),
                    show_progress_bar=False,
                )
                top = {
                    item
                    for item in sorted(
                        range(len(sentences)),
                        key=lambda item: (-float(scores[item]), item),
                    )[:3]
                }
                total += len(evidence.sentences)
                hits += sum(sentence in top for sentence in evidence.sentences)
        return hits / total if total else 0.0

    select.gold_sentence_rate = gold_sentence_rate
    return select


def select_best_retriever(
    validation_runs: Mapping[str, Mapping[int, Ranking]],
    validation_claims: Claims,
) -> str:
    """Select a supplied first-stage run using validation retrieval nDCG@10."""
    from .metrics import evaluate_retrieval, gold_documents

    if not validation_runs:
        raise ValueError("at least one validation retriever run is required")
    gold = gold_documents(validation_claims)
    scores = {
        name: float(
            evaluate_retrieval(
                {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in run.items()},
                gold,
            )["mean"]["ndcg@10"]
        )
        for name, run in validation_runs.items()
    }
    return max(scores, key=lambda name: (scores[name], name))


def validation_gold_triples(claims: Claims) -> set[Verification]:
    """Convert annotated validation evidence to checker-compatible triples."""
    return {
        (record.claim_id, item.doc_id, item.label)
        for record in claims.records
        for item in record.evidence
        if item.label in {"SUPPORT", "CONTRADICT"}
    }


def build_validation_errors(
    claims: Claims,
    predictions: Collection[Verification],
    rankings: Mapping[int, Ranking],
    corpus: Corpus,
) -> dict[int, str]:
    """Return categories only for validation claims with exact pair mismatches."""
    predicted_by_claim: dict[int, set[tuple[int, str]]] = {}
    for claim_id, doc_id, label in predictions:
        predicted_by_claim.setdefault(claim_id, set()).add((doc_id, label))
    errors: dict[int, str] = {}
    for record in claims.records:
        gold = {
            (item.doc_id, item.label)
            for item in record.evidence
            if item.label in {"SUPPORT", "CONTRADICT"}
        }
        predicted = predicted_by_claim.get(record.claim_id, set())
        if gold == predicted:
            continue
        retrieved = [doc_id for doc_id, _ in rankings.get(record.claim_id, [])[:5]]
        errors[record.claim_id] = classify_error(
            record,
            gold,
            predicted,
            retrieved,
            corpus,
        )
    return errors


def run_task5(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    validation_runs: Mapping[str, Mapping[int, Ranking]] | None = None,
    evaluation_runs: Mapping[str, Mapping[int, Ranking]] | None = None,
) -> dict[str, Any]:
    """Run verification from genuine supplied first-stage rankings.

    Rankings are explicit inputs so this stage cannot silently fabricate or
    regenerate retrieval candidates. Previous task runners can pass their
    validation and evaluation runs directly.
    """
    if validation_runs is None or evaluation_runs is None:
        raise ValueError(
            "run_task5 requires genuine validation_runs and evaluation_runs "
            "from the completed retrieval stages"
        )
    config = load_config(config_path)
    result_path = Path(config["paths"]["results_dir"]) / "task5_verification.json"
    summary_path = Path(config["paths"]["results_dir"]) / "task5_summary.json"
    prediction_dir = Path(config["paths"]["predictions_dir"])
    ft_enabled = bool(config["verification"].get("ft", {}).get("enabled", False))
    set_global_seed(int(config["seed"]))
    corpus = load_corpus(config["paths"]["corpus"])
    splits = load_all_claim_splits(config["paths"]["data_dir"])
    best_name = select_best_retriever(validation_runs, splits["val"])
    if best_name not in evaluation_runs:
        raise ValueError(f"evaluation rankings missing selected retriever: {best_name}")
    verification_config = config["verification"]
    threshold_grid = verification_threshold_grid(verification_config)
    base_settings = VerificationSettings(
        model_name=str(verification_config["model_name"]),
        batch_size=int(verification_config["batch_size"]),
        device=resolve_device(str(verification_config["device"])),
        candidate_depth=int(verification_config["candidate_depth"]),
        support_threshold=float(verification_config["support_threshold"]),
        contradict_threshold=float(verification_config["contradict_threshold"]),
        gate=float(verification_config["gate"]),
    )
    gold = validation_gold_triples(splits["val"])
    cache_dir = Path(config["paths"]["cache_dir"])
    base_backend = NLIBackend(base_settings)
    whole_premise = whole_abstract_premise
    dense_premise_selector = _build_premise_selector(config)
    dense_model_config = config["dense"]["models"]["general"]
    dense_selector_cache_key = json.dumps({
        "kind": "dense_sentence_selector",
        "model_name": dense_model_config["name"],
        "query_prefix": dense_model_config.get("query_prefix", ""),
        "max_seq_length": config["dense"]["max_seq_length"],
        "premise_sentences": verification_config["premise_sentences"],
        "premise_max_length": verification_config["premise_max_length"],
    }, sort_keys=True)
    whole_abstract_cache_key = "whole_abstract:v1"
    validation_cache, val_latency_ms = base_backend.score_cache(
        splits["val"],
        validation_runs[best_name],
        corpus,
        cache_dir / "nli_val.json",
        dense_premise_selector,
        cache_key=dense_selector_cache_key,
    )
    evaluation_cache, eval_latency_ms = base_backend.score_cache(
        splits["eval"],
        evaluation_runs[best_name],
        corpus,
        cache_dir / "nli_eval.json",
        dense_premise_selector,
        cache_key=dense_selector_cache_key,
    )
    whole_validation_cache, _ = base_backend.score_cache(
        splits["val"], validation_runs[best_name], corpus, cache_dir / "nli_val_whole.json", whole_premise,
        cache_key=whole_abstract_cache_key,
    )
    whole_evaluation_cache, _ = base_backend.score_cache(
        splits["eval"], evaluation_runs[best_name], corpus, cache_dir / "nli_eval_whole.json", whole_premise,
        cache_key=whole_abstract_cache_key,
    )

    variants: dict[str, tuple[NLIBackend, Any, dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]] = {
        "zero_shot_whole_abstract": (base_backend, whole_premise, whole_validation_cache, whole_evaluation_cache),
        "zero_shot_sentence_selector": (base_backend, dense_premise_selector, validation_cache, evaluation_cache),
    }
    selector_rate = 0.0
    if ft_enabled:
        ft_config = verification_config["ft"]
        selector_checkpoint = project_path(config, ft_config["selector_checkpoint"])
        nli_checkpoint = project_path(config, ft_config["nli_checkpoint"])
        selector = _build_selector_premise_builder(config, selector_checkpoint)
        ft_backend = NLIBackend(base_settings, nli_checkpoint)
        selector_stat = selector_checkpoint.stat()
        selector_cache_key = json.dumps({
            "kind": "fine_tuned_selector",
            "model_name": ft_config["selector_model_name"],
            "checkpoint": {
                "path": str(selector_checkpoint.resolve()),
                "size": selector_stat.st_size,
                "mtime_ns": selector_stat.st_mtime_ns,
            },
            "premise_sentences": verification_config["premise_sentences"],
            "premise_max_length": verification_config["premise_max_length"],
        }, sort_keys=True)
        selector_val_cache, _ = base_backend.score_cache(
            splits["val"], validation_runs[best_name], corpus,
            cache_dir / "nli_ft_selector_val.json", selector,
            cache_key=selector_cache_key,
        )
        selector_eval_cache, _ = base_backend.score_cache(
            splits["eval"], evaluation_runs[best_name], corpus,
            cache_dir / "nli_ft_selector_eval.json", selector,
            cache_key=selector_cache_key,
        )
        both_val_cache, _ = ft_backend.score_cache(
            splits["val"], validation_runs[best_name], corpus,
            cache_dir / "nli_ft_both_val.json", selector,
            cache_key=selector_cache_key,
        )
        both_eval_cache, _ = ft_backend.score_cache(
            splits["eval"], evaluation_runs[best_name], corpus,
            cache_dir / "nli_ft_both_eval.json", selector,
            cache_key=selector_cache_key,
        )
        nli_val_cache, _ = ft_backend.score_cache(
            splits["val"], validation_runs[best_name], corpus,
            cache_dir / "nli_ft_val.json", dense_premise_selector,
            cache_key=dense_selector_cache_key,
        )
        nli_eval_cache, _ = ft_backend.score_cache(
            splits["eval"], evaluation_runs[best_name], corpus,
            cache_dir / "nli_ft_eval.json", dense_premise_selector,
            cache_key=dense_selector_cache_key,
        )
        variants["fine_tuned_selector"] = (
            base_backend, selector, selector_val_cache, selector_eval_cache,
        )
        variants["fine_tuned_nli"] = (
            ft_backend, dense_premise_selector, nli_val_cache, nli_eval_cache,
        )
        variants["both"] = (
            ft_backend, selector, both_val_cache, both_eval_cache,
        )
        selector_rate = float(selector.gold_sentence_rate(splits["val"], corpus))

    def evaluate_setting(
        backend: NLIBackend,
        cache: Mapping[int, Sequence[Mapping[str, Any]]],
        support: float,
        contradict: float,
        gate: float,
        depth: int,
    ) -> dict[str, float]:
        settings = VerificationSettings(**{
            **base_settings.__dict__,
            "support_threshold": support,
            "contradict_threshold": contradict,
            "gate": gate,
            "candidate_depth": depth,
        })
        backend.settings = settings
        predictions = backend.from_cache(splits["val"], cache)
        metrics = verification_metrics(predictions, gold)
        metrics["prediction_count"] = float(len(predictions))
        metrics["support_prediction_count"] = float(
            sum(label == "SUPPORT" for _, _, label in predictions)
        )
        metrics["contradict_prediction_count"] = float(
            sum(label == "CONTRADICT" for _, _, label in predictions)
        )
        return metrics

    variant_scores: dict[str, float] = {}
    selected_by_variant: dict[str, tuple[float, ...]] = {}
    grid_by_variant: dict[str, dict[str, Any]] = {}
    for variant_name, (variant_backend, _, variant_val_cache, _) in variants.items():
        selected_variant, variant_grid = select_thresholds(
            threshold_grid,
            lambda *setting: evaluate_setting(
                variant_backend, variant_val_cache, *setting
            ),
            prefer_coverage_on_ties=variant_name in {"fine_tuned_nli", "both"},
        )
        selected_by_variant[variant_name] = selected_variant
        grid_by_variant[variant_name] = variant_grid
        variant_scores[variant_name] = float(
            variant_grid[":".join(map(str, selected_variant))]["f1"]
        )
    selected_variant_name = max(
        variant_scores,
        key=lambda name: (variant_scores[name], name),
    )
    nli_validation_comparison = compare_nli_validation_f1(variant_scores)
    selected = selected_by_variant[selected_variant_name]
    backend, _, validation_cache, evaluation_cache = variants[selected_variant_name]
    grid_results = grid_by_variant[selected_variant_name]
    backend.settings = VerificationSettings(**{
        **base_settings.__dict__,
        "support_threshold": selected[0],
        "contradict_threshold": selected[1],
        "gate": selected[2],
        "candidate_depth": selected[3],
    })
    val_predictions = backend.from_cache(splits["val"], validation_cache)
    eval_predictions = backend.from_cache(splits["eval"], evaluation_cache)
    prediction_dir = Path(config["paths"]["predictions_dir"])
    write_verification_csv(prediction_dir / "t5_val_verification.csv", val_predictions)
    write_verification_csv(prediction_dir / "t5_verification.csv", eval_predictions)
    errors = build_validation_errors(
        splits["val"], val_predictions, validation_runs[best_name], corpus
    )
    write_error_csv(prediction_dir / "t5_val_errors.csv", errors)
    result = {
        "selected_retriever": best_name,
        "model_name": base_settings.model_name,
        "parameter_count": backend.parameter_count,
        "selected_variant": selected_variant_name,
        "variant_f1": variant_scores,
        "fine_tuned_nli_vs_whole_abstract": nli_validation_comparison,
        "variant_selection": {
            name: {
                "thresholds": {
                    "support": selected_settings[0],
                    "contradict": selected_settings[1],
                    "gate": selected_settings[2],
                    "candidate_depth": selected_settings[3],
                },
                "validation_metrics": grid_by_variant[name][
                    ":".join(map(str, selected_settings))
                ],
            }
            for name, selected_settings in selected_by_variant.items()
        },
        "selector_val_gold_sentence_top3_rate": selector_rate,
        "selected_thresholds": {
            "support": selected[0],
            "contradict": selected[1],
            "gate": selected[2],
            "candidate_depth": selected[3],
        },
        "validation_metrics": grid_results[":".join(map(str, selected))],
        "tested_settings": grid_results,
        "validation_inference_latency_ms": val_latency_ms,
        "evaluation_inference_latency_ms": eval_latency_ms,
    }
    result_path = Path(config["paths"]["results_dir"]) / "task5_verification.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps({
        "variant_f1": variant_scores,
        "fine_tuned_nli_vs_whole_abstract": nli_validation_comparison,
        "selector_val_gold_sentence_top3_rate": selector_rate,
        "selected_variant": selected_variant_name,
    }, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    config = load_config(DEFAULT_CONFIG)
    splits = load_all_claim_splits(config["paths"]["data_dir"])
    corpus = load_corpus(config["paths"]["corpus"])
    prediction_dir = Path(config["paths"]["predictions_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    doc_ids = {record.doc_id for record in corpus.records}
    validation_runs = {
        "t3_rerank": read_trec(
            results_dir / "val_rerank.trec",
            expected_claim_ids={record.claim_id for record in splits["val"].records},
            expected_doc_ids=doc_ids,
        ),
    }
    evaluation_runs = {
        "t3_rerank": read_trec(
            prediction_dir / "t3_rerank.trec",
            expected_claim_ids={record.claim_id for record in splits["eval"].records},
            expected_doc_ids=doc_ids,
        ),
    }
    run_task5(
        validation_runs=validation_runs,
        evaluation_runs=evaluation_runs,
    )
