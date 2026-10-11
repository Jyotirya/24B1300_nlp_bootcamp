"""Dependency-ordered orchestration for the complete SciFact pipeline."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Mapping

# Allow ``python -m pipeline`` when the current directory is ``code/scripts/``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "scripts"

from .common import DEFAULT_CONFIG, load_config
from .data import load_all_claim_splits, load_corpus
from .metrics import read_trec
from .fuse import build_extra_hybrid_ft, fuse_runs
from .retrieve import run_task1
from .dense import run_task2
from .rerank import run_task3
from .finetune import run_task4
from .verify import run_task5
from .report import write_report
from .latency import measure_latency


def _read_stage_runs(config: Mapping[str, Any]) -> tuple[
    dict[str, dict[int, list[tuple[int, float]]]],
    dict[str, dict[int, list[tuple[int, float]]]],
]:
    """Read genuine validation/evaluation runs produced by Tasks 1 and 2."""
    paths = config["paths"]
    data_dir = Path(paths["data_dir"])
    splits = load_all_claim_splits(data_dir)
    doc_ids = {record.doc_id for record in load_corpus(paths["corpus"]).records}
    prediction_dir = Path(paths["predictions_dir"])
    results_dir = Path(paths["results_dir"])
    validation: dict[str, dict[int, list[tuple[int, float]]]] = {}
    evaluation: dict[str, dict[int, list[tuple[int, float]]]] = {}
    for name, eval_file, val_file in (
        ("bm25", "t1_bm25.trec", "val_bm25.trec"),
        ("dense_general", "t2_general.trec", "val_general.trec"),
        ("dense_scientific", "t2_scientific.trec", "../results/val_scientific.trec"),
    ):
        evaluation[name] = read_trec(
            prediction_dir / eval_file,
            expected_claim_ids={record.claim_id for record in splits["eval"].records},
            expected_doc_ids=doc_ids,
        )
        validation_path = (
            results_dir / val_file.removeprefix("../results/")
            if val_file.startswith("../results/")
            else prediction_dir / val_file
        )
        validation[name] = read_trec(
            validation_path,
            expected_claim_ids={record.claim_id for record in splits["val"].records},
            expected_doc_ids=doc_ids,
        )
    return validation, evaluation


def _read_trec_if_present(
    path: Path,
    claim_ids: set[int],
    doc_ids: set[int],
) -> dict[int, list[tuple[int, float]]]:
    if not path.is_file():
        return {}
    return read_trec(path, expected_claim_ids=claim_ids, expected_doc_ids=doc_ids)


def run_pipeline(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Execute Tasks 1-5 in dependency order without fabricating stage inputs."""
    config_path = Path(config_path)
    run_task1(config_path)
    run_task2(config_path)
    config = load_config(config_path)
    splits = load_all_claim_splits(config["paths"]["data_dir"])
    doc_ids = {record.doc_id for record in load_corpus(config["paths"]["corpus"]).records}
    validation_runs, evaluation_runs = _read_stage_runs(config)
    task3 = run_task3(
        config_path,
        retrieval_runs={
            "validation": validation_runs,
            "evaluation": evaluation_runs,
        },
    )
    task4 = run_task4(config_path)
    validation_ids = {record.claim_id for record in splits["val"].records}
    evaluation_ids = {record.claim_id for record in splits["eval"].records}
    prediction_dir = Path(config["paths"]["predictions_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    for name, val_file, eval_file in (
        ("t3_rerank", "../results/val_rerank.trec", "t3_rerank.trec"),
        ("t3_hybrid", "val_hybrid.trec", "t3_hybrid.trec"),
        ("t4_random", "val_t4_random.trec", "t4_random.trec"),
        ("t4_hard", "val_t4_hard.trec", "t4_hard.trec"),
    ):
        validation_path = (
            results_dir / val_file.removeprefix("../results/")
            if val_file.startswith("../results/")
            else prediction_dir / val_file
        )
        evaluation_path = prediction_dir / eval_file
        validation_run = _read_trec_if_present(validation_path, validation_ids, doc_ids)
        evaluation_run = _read_trec_if_present(evaluation_path, evaluation_ids, doc_ids)
        if validation_run and evaluation_run:
            validation_runs[name] = validation_run
            evaluation_runs[name] = evaluation_run
    extra_validation, extra_evaluation, extra_summary = build_extra_hybrid_ft(
        config,
        validation_runs,
        evaluation_runs,
        splits["val"],
        splits["eval"],
    )
    validation_runs["extra_hybrid_ft"] = extra_validation
    evaluation_runs["extra_hybrid_ft"] = extra_evaluation
    task3_summary_path = results_dir / "task3_summary.json"
    if task3_summary_path.is_file():
        task3_summary = json.loads(task3_summary_path.read_text(encoding="utf-8"))
        task3_summary["extra_hybrid_ft"] = extra_summary
        task3_summary_path.write_text(
            json.dumps(task3_summary, indent=2) + "\n",
            encoding="utf-8",
        )
    if "t3_hybrid" not in validation_runs:
        validation_runs["t3_hybrid"] = fuse_runs(
            validation_runs["bm25"],
            validation_runs["dense_general"],
            scientific_dense_run=validation_runs["dense_scientific"],
            method="rrf",
            depth=int(config["fusion"]["candidate_depth"]),
            rrf_constant=int(config["fusion"]["rrf_constant"]),
        )
        evaluation_runs["t3_hybrid"] = fuse_runs(
            evaluation_runs["bm25"],
            evaluation_runs["dense_general"],
            scientific_dense_run=evaluation_runs["dense_scientific"],
            method="rrf",
            depth=int(config["fusion"]["candidate_depth"]),
            rrf_constant=int(config["fusion"]["rrf_constant"]),
        )
    task5 = run_task5(
        config_path,
        validation_runs=validation_runs,
        evaluation_runs=evaluation_runs,
    )
    latency_path = Path(config["paths"]["results_dir"]) / "latency.json"
    if not latency_path.is_file():
        measure_latency(config_path)
    write_report(config_path)
    return {"task3": task3, "task4": task4, "task5": task5}


if __name__ == "__main__":
    run_pipeline()
