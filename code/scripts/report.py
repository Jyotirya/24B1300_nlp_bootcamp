"""Build the submission results summary and concise task report."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .common import DEFAULT_CONFIG, load_config
from .data import load_all_claim_splits, load_corpus
from .metrics import evaluate_retrieval, read_trec
from .verify import verification_metrics


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required results file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"results file must contain an object: {path}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric result for {name}")
    return float(value)


def _under_100_words(text: str) -> str:
    words = text.split()
    return " ".join(words[:100])


def _template(config: Mapping[str, Any]) -> dict[str, Any]:
    configured = Path(config["paths"]["results_template"])
    candidates = [configured, Path(config["paths"]["root"]) / "results_template.json"]
    for path in candidates:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("results template must be a JSON object")
            return value
    raise FileNotFoundError("results_template.json was not found")


def _run(path: Path, claims: set[int], docs: set[int]) -> dict[int, list[tuple[int, float]]]:
    return read_trec(path, expected_claim_ids=claims, expected_doc_ids=docs)


def _select_q1_claims(gaps: Mapping[int, float]) -> tuple[list[int], list[int]]:
    """Select two evidence-bearing claims for each direction of the Q1 comparison."""
    if len(gaps) < 2:
        raise ValueError(
            "Q1 report requires at least two validation claims with evidence"
        )
    bm25_wins = sorted(gaps, key=lambda cid: (-gaps[cid], cid))[:2]
    dense_wins = sorted(gaps, key=lambda cid: (gaps[cid], cid))[:2]
    return bm25_wins, dense_wins


def _normalize_verification_csv(path: Path) -> tuple[list[tuple[int, int, str]], int]:
    """Remove duplicate pairs; omit conflicting labels when scores are unavailable."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["claim_id", "doc_id", "label"]:
            raise ValueError(f"invalid verification CSV header: {path}")
        labels_by_pair: dict[tuple[int, int], set[str]] = {}
        for line_number, row in enumerate(reader, 2):
            if len(row) != 3:
                raise ValueError(f"{path}:{line_number} must contain three columns")
            try:
                claim_id, doc_id = int(row[0]), int(row[1])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} claim_id and doc_id must be integers"
                ) from exc
            label = row[2].strip().upper()
            if label not in {"SUPPORT", "CONTRADICT"}:
                raise ValueError(f"{path}:{line_number} has invalid label {row[2]!r}")
            labels_by_pair.setdefault((claim_id, doc_id), set()).add(label)

    conflicts = sum(len(labels) > 1 for labels in labels_by_pair.values())
    normalized = [
        (claim_id, doc_id, next(iter(labels)))
        for (claim_id, doc_id), labels in labels_by_pair.items()
        if len(labels) == 1
    ]
    normalized.sort()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["claim_id", "doc_id", "label"])
        writer.writerows(normalized)
    return normalized, conflicts


def _q1(
    config: Mapping[str, Any],
) -> tuple[list[int], list[int], str]:
    data_dir = Path(config["paths"]["data_dir"])
    corpus = load_corpus(config["paths"]["corpus"])
    splits = load_all_claim_splits(data_dir)
    claims = {record.claim_id: record for record in splits["val"].records}
    doc_ids = {record.doc_id for record in corpus.records}
    prediction_dir = Path(config["paths"]["predictions_dir"])
    bm25 = _run(prediction_dir / "val_bm25.trec", set(claims), doc_ids)
    dense = _run(prediction_dir / "val_general.trec", set(claims), doc_ids)
    gold = {claim_id: {item.doc_id for item in record.evidence} for claim_id, record in claims.items()}
    bm25_metrics = evaluate_retrieval(
        {cid: [doc for doc, _ in ranking] for cid, ranking in bm25.items()}, gold
    )["per_claim"]
    dense_metrics = evaluate_retrieval(
        {cid: [doc for doc, _ in ranking] for cid, ranking in dense.items()}, gold
    )["per_claim"]
    gaps = {
        cid: float(bm25_metrics[cid]["ndcg@10"]) - float(dense_metrics[cid]["ndcg@10"])
        for cid in claims if gold[cid]
    }
    bm25_wins, dense_wins = _select_q1_claims(gaps)

    def detail(ids: list[int]) -> str:
        lines = []
        for claim_id in ids:
            record = claims[claim_id]
            lines.append(
                f"{claim_id}: BM25 {bm25_metrics[claim_id]['ndcg@10']:.4f}, "
                f"dense {dense_metrics[claim_id]['ndcg@10']:.4f}; "
                f"claim=\"{record.claim}\""
            )
        return "\n".join(lines)

    text = (
        "Q1. Largest BM25-minus-dense gaps:\n" + detail(bm25_wins) +
        "\nLargest dense-minus-BM25 gaps:\n" + detail(dense_wins) +
        "\nExplanation: TODO (write by hand)"
    )
    return bm25_wins, dense_wins, text


def _q2(results_dir: Path) -> str:
    summary = _json(results_dir / "task3_summary.json")
    latency = _json(results_dir / "latency.json")
    gains = summary["ndcg_gain_over_previous"]
    component = max(
        (name for name in gains if name not in {"bm25", "dense"} and gains[name] is not None),
        key=lambda name: float(gains[name]),
    )
    latency_key = {
        "bm25": "bm25_ms_per_query",
        "dense": "general_dense_ms_per_query",
        "hybrid": "hybrid_ms_per_query",
        "rerank": "rerank_ms_per_query",
    }.get(component)
    if latency_key is None or latency_key not in latency:
        raise ValueError(f"missing latency for Task 3 component {component}")
    return (
        f"Q2. The largest Task 3 validation nDCG gain was from {component}, "
        f"with gain {_number(gains[component], component):.4f} and "
        f"latency {_number(latency[latency_key], f'{component} latency'):.3f} ms/query."
    )


def _q3(results_dir: Path) -> str:
    curves = _json(results_dir / "task4_curves.json")
    base = _number(curves["base"]["val_ndcg@10"], "base validation nDCG@10")

    def epoch_entries(name: str) -> list[tuple[int, Mapping[str, Any]]]:
        raw = curves.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"expected an object of epoch results for {name}")
        entries: list[tuple[int, Mapping[str, Any]]] = []
        for key, value in raw.items():
            # task4_curves.json also stores metadata such as epochs_trained
            # and best_epoch; only numeric keys are actual epoch records.
            if str(key).isdigit() and isinstance(value, dict):
                entries.append((int(key), value))
        if not entries:
            raise ValueError(f"no epoch records found for {name} in task4_curves.json")
        return sorted(entries, key=lambda item: item[0])

    def run_value(name: str) -> tuple[float, str]:
        entries = epoch_entries(name)
        best_epoch, best_values = max(
            entries,
            key=lambda item: _number(item[1].get("val_ndcg@10"), f"{name} epoch {item[0]} validation nDCG@10"),
        )
        best = _number(best_values.get("val_ndcg@10"), f"{name} validation nDCG@10")
        curve_parts = []
        for epoch, values in entries:
            train_loss = _number(values.get("train_loss"), f"{name} epoch {epoch} train loss")
            val_score = _number(values.get("val_ndcg@10"), f"{name} epoch {epoch} validation nDCG@10")
            curve_parts.append(f"e{epoch}: loss={train_loss:.4f}, val={val_score:.4f}")
        curve = ", ".join(curve_parts)
        return best, f"{name} best=e{best_epoch} ({best:.4f}; {curve})"

    _, random_text = run_value("t4_random")
    _, hard_text = run_value("t4_hard")
    signal = "no clear overfit signal"
    for name in ("t4_random", "t4_hard"):
        entries = epoch_entries(name)
        for (e1, v1), (e2, v2) in zip(entries, entries[1:]):
            val1 = _number(v1.get("val_ndcg@10"), f"{name} epoch {e1} validation nDCG@10")
            val2 = _number(v2.get("val_ndcg@10"), f"{name} epoch {e2} validation nDCG@10")
            loss1 = _number(v1.get("train_loss"), f"{name} epoch {e1} train loss")
            loss2 = _number(v2.get("train_loss"), f"{name} epoch {e2} train loss")
            if val2 < val1:
                signal = f"{name} validation dropped at epoch {e2}"
                break
            if loss2 < loss1 and val2 <= val1:
                signal = f"{name} loss kept falling while validation stalled at epoch {e2}"
                break
        if signal != "no clear overfit signal":
            break
    return _under_100_words(
        f"Q3. Base={base:.4f}; {random_text}; {hard_text}. Signal: {signal}."
    )


def _q4(prediction_dir: Path) -> str:
    path = prediction_dir / "t5_val_errors.csv"
    if not path.is_file():
        raise FileNotFoundError(f"required error file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    total = len(rows)
    misses = sum(row["category"] == "retrieval_miss" for row in rows)
    return _under_100_words(f"Q4. Retrieval misses: {misses}/{total} validation errors.")


def write_report(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    root = Path(config["paths"]["root"])
    results_dir = Path(config["paths"]["results_dir"])
    prediction_dir = Path(config["paths"]["predictions_dir"])
    template = _template(config)
    task1 = _json(results_dir / "task1_bm25_tuning.json")
    task2 = _json(results_dir / "task2_dense.json")
    task3 = _json(results_dir / "task3_hybrid_rerank.json")
    task4 = _json(results_dir / "task4_finetuning.json")
    task5 = _json(results_dir / "task5_verification.json")
    q1_bm25, q1_dense, q1_text = _q1(config)
    q2_text = _q2(results_dir)
    q3_text = _q3(results_dir)
    q4_text = _q4(prediction_dir)

    general = task2["models"]["general"]
    scientific = task2["models"]["scientific"]
    rerank = task3["rerank"]
    import torch
    filled = json.loads(json.dumps(template))
    filled["seed"] = int(config["seed"])
    filled["gpu_used"] = bool(torch.cuda.is_available())
    filled["task1"] = {
        "bm25_settings": json.dumps(task1["selected_settings"], sort_keys=True),
        "val_ndcg10": _number(task1["best_validation_metrics"]["ndcg@10"], "BM25 nDCG"),
    }
    filled["task2"] = {
        "general_model": general["settings"]["name"],
        "scientific_model": scientific["settings"]["name"],
        "val_ndcg10_general": _number(general["validation_metrics"]["ndcg@10"], "general nDCG"),
        "val_ndcg10_scientific": _number(scientific["validation_metrics"]["ndcg@10"], "scientific nDCG"),
        "index_time_sec": {
            "general": _number(general["indexing_time_sec"], "general index time"),
            "scientific": _number(scientific["indexing_time_sec"], "scientific index time"),
        },
        "query_latency_ms": {
            "general": _number(general["query_latency"]["mean_query_latency_ms"], "general latency"),
            "scientific": _number(scientific["query_latency"]["mean_query_latency_ms"], "scientific latency"),
        },
        "memory_mb": {
            "general": _number(general["memory_delta_mb"], "general memory"),
            "scientific": _number(scientific["memory_delta_mb"], "scientific memory"),
        },
    }
    filled["task3"] = {
        "hybrid_method": json.dumps(task3["fusion"]["selected"], sort_keys=True),
        "reranker_model": rerank["model_name"],
        "rerank_top_k": int(rerank["selected_k"]),
        "val_ndcg10_hybrid": _number(task3["fusion"]["first_stage_metrics"]["hybrid"]["ndcg@10"], "hybrid nDCG"),
        "val_ndcg10_rerank": _number(rerank["validation_metrics"]["ndcg@10"], "rerank nDCG"),
        "rerank_query_latency_ms": _number(rerank["validation_latency"]["mean_query_latency_ms"], "rerank latency"),
    }
    curves = _json(results_dir / "task4_curves.json")
    def best_epoch_score(name: str) -> float:
        raw = curves.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"expected an object of epoch results for {name}")
        scores = [
            _number(value.get("val_ndcg@10"), f"{name} validation nDCG@10")
            for key, value in raw.items()
            if str(key).isdigit() and isinstance(value, dict)
        ]
        if not scores:
            raise ValueError(f"no numeric epoch results found for {name} in task4_curves.json")
        return max(scores)

    filled["task4"] = {
        "base_model": task4["base_model"],
        "val_ndcg10_random_negatives": best_epoch_score("t4_random"),
        "val_ndcg10_hard_negatives": best_epoch_score("t4_hard"),
    }
    variant_name = str(task5.get("selected_variant", "zero_shot_whole_abstract"))
    for filename in ("t5_verification.csv", "t5_val_verification.csv"):
        verification_path = prediction_dir / filename
        if verification_path.is_file():
            _, conflicts = _normalize_verification_csv(verification_path)
            if conflicts:
                print(
                    f"Removed {conflicts} conflicting claim/document pairs "
                    f"from {filename}; CSV scores are unavailable to choose a label."
                )
    val_predictions_path = prediction_dir / "t5_val_verification.csv"
    if val_predictions_path.is_file():
        with val_predictions_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        val_predictions = {
            (int(row["claim_id"]), int(row["doc_id"]), row["label"].upper())
            for row in rows
        }
        validation_gold = {
            (record.claim_id, evidence.doc_id, evidence.label)
            for record in load_all_claim_splits(config["paths"]["data_dir"])["val"].records
            for evidence in record.evidence
            if evidence.label in {"SUPPORT", "CONTRADICT"}
        }
        task5_val_f1 = verification_metrics(
            val_predictions, validation_gold
        )["f1"]
    else:
        task5_val_f1 = _number(
            task5["validation_metrics"]["f1"], "Task 5 F1"
        )
    filled["task5"] = {
        "retriever_used": task5["selected_retriever"],
        "nli_model": task5["model_name"],
        "zero_shot_or_finetuned": "fine-tuned" if "fine_tuned" in variant_name or "both" in variant_name else "zero-shot",
        "val_f1": task5_val_f1,
    }
    filled["report_q1"] = {"bm25_wins": q1_bm25, "general_wins": q1_dense}
    filled["roll_no"] = str(config["roll_no"])
    if filled["name"] == "":
        raise ValueError("cannot fill template key: name")
    if filled["roll_no"] == "":
        raise ValueError("cannot fill template key: roll_no")
    expected_keys = set(template)
    if set(filled) != expected_keys:
        raise ValueError("results.json keys do not exactly match results_template.json")
    (root / "results.json").write_text(json.dumps(filled, indent=2) + "\n", encoding="utf-8")
    report = "\n\n".join([q1_text, q2_text, q3_text, q4_text, "Q5. TODO: add your qualitative analysis."])
    (root / "report.md").write_text(report + "\n", encoding="utf-8")
    checker_candidates = (
        Path(config["paths"]["data_dir"]) / "check_format.py",
        root / "check_format.py",
    )
    checker = next((path for path in checker_candidates if path.is_file()), None)
    if checker is None:
        print("WARNING: check_format.py was not found")
        return filled
    completed = subprocess.run(
        [sys.executable, str(checker), str(root)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return filled


if __name__ == "__main__":
    write_report()
