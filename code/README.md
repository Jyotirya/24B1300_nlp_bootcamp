# SciFact NLP Bootcamp Reproduction Guide

This repository reproduces the SciFact task described in `NLP_PS.pdf`: build a retrieval pipeline for scientific claim verification, compare lexical and dense retrievers, hybridize and rerank them, fine-tune a bi-encoder, and then verify claims with an NLI-style classifier.

## Project layout

- `data/` — SciFact corpus and claim splits
  - `corpus.jsonl`
  - `train_claims.jsonl`
  - `val_claims.jsonl`
  - `eval_claims.jsonl`
- `code/scripts/` — task implementations and helpers
- `predictions/` — TREC run files and verification outputs
- `results.json` — filled submission template
- `report.md` — answers to Q1–Q5
- `check_format.py` — validates file names and formats
- `make_zip.py` — packages the final submission zip

## Requirements

- Python 3.10+
- CUDA is optional; the code automatically falls back to CPU if no GPU is available
- Internet access for downloading pretrained Hugging Face models

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r code/scripts/requirements.txt
export PYTHONPATH=$PWD/code
```

## End-to-end reproduction

To reproduce the full pipeline in dependency order (Task 1 → Task 5), run:

```bash
PYTHONPATH=code python3 -m scripts.pipeline
```

This step generates the required runs and summary files, including:

- `predictions/t1_bm25.trec`
- `predictions/t2_general.trec`
- `predictions/t2_scientific.trec`
- `predictions/t3_hybrid.trec`
- `predictions/t3_rerank.trec`
- `predictions/t4_random.trec`
- `predictions/t4_hard.trec`
- `predictions/t5_verification.csv`
- `predictions/t5_val_verification.csv`
- `predictions/t5_val_errors.csv`
- `results.json`
- `report.md`
- `results/latency.json`
- `results/task3_summary.json`
- `results/task4_curves.json`

The `pipeline` script calls the task implementations in order and writes the final submission metadata.

## Run individual tasks

If you want to run each stage manually:

```bash
PYTHONPATH=code python3 -m scripts.retrieve   # Task 1: BM25 baseline
PYTHONPATH=code python3 -m scripts.dense     # Task 2: dense retrieval
PYTHONPATH=code python3 -m scripts.rerank    # Task 3: hybrid + rerank
PYTHONPATH=code python3 -m scripts.finetune  # Task 4: domain fine-tuning
PYTHONPATH=code python3 -m scripts.verify    # Task 5: verification + validation errors
PYTHONPATH=code python3 -m scripts.report    # Builds results.json and report.md
```

## Validation and packaging

Before submitting, validate the generated outputs against the checker:

```bash
python3 check_format.py .
```

or validate a zipped submission:

```bash
python3 check_format.py <rollno>_nlp_bootcamp.zip
```

To create the final submission archive:

```bash
PYTHONPATH=code python3 make_zip.py
```

This writes a zip named like `<rollno>_nlp_bootcamp.zip` containing:

- `code/`
- `predictions/`
- `results.json`
- `report.md`

## Task requirements from the problem statement

### Task 1 — Lexical baseline
- Explore the corpus and claims.
- Implement BM25 and tune it on the validation split.
- Output: `t1_bm25.trec`

### Task 2 — Dense retrieval
- Retrieve with two pretrained encoders: one general-purpose and one scientific/biomedical.
- Record index time, query latency and memory use in `results.json`.
- Output: `t2_general.trec`, `t2_scientific.trec`

### Task 3 — Combine and rerank
- Build a hybrid BM25 + dense system.
- Rerank the best system’s top-k with a cross-encoder.
- Output: `t3_hybrid.trec`, `t3_rerank.trec`

### Task 4 — Domain fine-tuning
- Fine-tune a bi-encoder twice on training claims:
  - random negatives
  - hard negatives mined from Tasks 1–2
- Output: `t4_random.trec`, `t4_hard.trec`

### Task 5 — Claim verification and error analysis
- For each claim, predict SUPPORT / CONTRADICT / NOT ENOUGH INFO using the best retriever and an NLI-style classifier.
- NLI fine-tuning uses the same dense sentence-selector premises as inference, optionally mixed with gold rationale examples; `results/task5_verification.json` reports its validation F1 against the zero-shot whole-abstract baseline.
- Submit predictions for validation and evaluation claims.
- Output: `t5_verification.csv`, `t5_val_verification.csv`, `t5_val_errors.csv`

## TREC format

Each retrieval run file must contain all claims for the relevant split with exactly 100 unique documents per claim, ranked 1..100:

```text
claim_id Q0 doc_id rank score run_name
```

The project scripts write these files in the required format automatically.

## Report requirements

After running the pipeline, `report.md` should contain short answers for Q1–Q5 (each answer under 100 words), and `results.json` should be filled using the template in `results_template.json`.

## Troubleshooting

- If a dependency is missing, reinstall the requirements file:
  ```bash
  pip install -r code/scripts/requirements.txt
  ```
- If Hugging Face models fail to download, verify that you have a working internet connection and enough disk space.
- If you see a format validation error, run `python3 check_format.py .` and fix the offending file according to the reported issue.
