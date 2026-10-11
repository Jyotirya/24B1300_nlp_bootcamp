import json
import inspect
import random
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.common import load_config, load_results_template, set_global_seed
from scripts.data import (
    ClaimRecord,
    Claims,
    Corpus,
    CorpusRecord,
    load_all_claim_splits,
    load_claims,
    load_corpus,
)
from scripts.metrics import evaluate_retrieval, ndcg_at_k, read_trec, recall_at_k, reciprocal_rank, write_trec
from scripts.retrieve import BM25Retriever, BM25Settings, rank_document_indices, tokenize_text
from scripts.dense import (
    DenseSettings,
    apply_query_prefix,
    count_model_parameters,
    normalize_embeddings,
    rank_embeddings,
)
from scripts.timing import latency_statistics, percentile
from scripts.fuse import apply_hybrid, interpolate_fuse, rrf_fuse, search_hybrid
from scripts.rerank import count_cross_encoder_parameters, rerank_one, select_candidate_k
from scripts.finetune import (
    FineTuneSettings,
    batch_without_duplicates,
    build_positive_pairs,
    mine_hard_negatives,
    select_best_checkpoint,
)
from scripts.finetune import _load_training_run
import scripts.finetune as finetune_module
from scripts.nli_ft import _selector_premise_builder, _training_examples
from scripts.verify import (
    NLIBackend,
    VerificationSettings,
    _best_verification_label,
    compare_nli_validation_f1,
    count_nli_parameters,
    limit_predictions,
    read_label_mapping,
    _select_premise_indices,
    select_thresholds,
    verification_metrics,
    verification_threshold_grid,
    write_verification_csv,
)
from scripts.report import _normalize_verification_csv, _select_q1_claims
from scripts.errors import ALLOWED_CATEGORIES, has_negation_cue
from scripts.pipeline import run_pipeline
from scripts.rerank import run_task3


ROOT = Path(__file__).resolve().parents[3]  # project root (contains data/, code/)


class FoundationSanityTests(unittest.TestCase):
    def test_corpus_preserves_ids_and_sentence_lists(self) -> None:
        corpus = load_corpus(ROOT / "data/corpus.jsonl")
        self.assertEqual(corpus.records[0].doc_id, 4983)
        self.assertEqual(corpus.id_to_index[4983], 0)
        self.assertIsInstance(corpus.records[0].abstract, list)
        self.assertTrue(all(isinstance(sentence, str) for sentence in corpus.records[0].abstract))

    def test_empty_evidence_is_preserved(self) -> None:
        claims = load_claims(ROOT / "data/train_claims.jsonl")
        self.assertEqual(claims.records[0].claim_id, 0)
        self.assertEqual(claims.records[0].evidence, [])
        self.assertEqual(claims.id_to_index[0], 0)

    def test_all_claim_splits_load(self) -> None:
        splits = load_all_claim_splits(ROOT / "data")
        self.assertEqual({name: len(value.records) for name, value in splits.items()},
                         {"train": 647, "val": 162, "eval": 300})
        self.assertEqual(splits["eval"].records[0].evidence, [])

    def test_results_template_schema_is_loaded_unchanged(self) -> None:
        loaded = load_results_template(ROOT / "results_template.json")
        with (ROOT / "results_template.json").open(encoding="utf-8") as handle:
            self.assertEqual(loaded, json.load(handle))

    def test_seed_is_deterministic(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            set_global_seed(42)
            first = (random.random(), np.random.random())
            set_global_seed(42)
            second = (random.random(), np.random.random())
        self.assertEqual(first, second)

    def test_duplicate_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claims.jsonl"
            row = {"id": 1, "claim": "x", "evidence": []}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate claim id"):
                load_claims(path)

    def test_retrieval_metrics(self) -> None:
        self.assertEqual(ndcg_at_k([7], {7}), 1.0)
        self.assertEqual(recall_at_k([7], {7}), 1.0)
        self.assertEqual(reciprocal_rank([7], {7}), 1.0)
        self.assertAlmostEqual(ndcg_at_k([8, 7], {7}), 1 / np.log2(3))
        self.assertEqual(reciprocal_rank([8, 7], {7}), 0.5)
        self.assertEqual(evaluate_retrieval({1: [8], 2: []}, {1: {7}, 2: set()})["mean"],
                         {"ndcg@10": 0.0, "recall@100": 0.0, "mrr": 0.0})
        multiple = evaluate_retrieval({1: [8, 7]}, {1: {7, 8}})
        self.assertEqual(multiple["per_claim"][1]["recall@100"], 1.0)

    def test_trec_round_trip_and_validation(self) -> None:
        ranking = {12: [(doc_id, float(100 - doc_id)) for doc_id in range(100)]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.trec"
            write_trec(path, ranking, "unit")
            loaded = read_trec(path, expected_claim_ids={12}, expected_doc_ids=set(range(100)))
            self.assertEqual([doc for doc, _ in loaded[12]], list(range(100)))
            scores = [score for _, score in loaded[12]]
            self.assertTrue(all(left > right for left, right in zip(scores, scores[1:])))
            with self.assertRaisesRegex(ValueError, "exactly 100"):
                write_trec(path, {12: [(1, 1.0)]}, "unit")
            duplicate = [(doc_id, 1.0) for doc_id in range(99)] + [(0, 1.0)]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                write_trec(path, {12: duplicate}, "unit")

    def test_bm25_tokenization_and_configuration(self) -> None:
        tokens = tokenize_text("IL-6 binds p53 and other hyphenated-terms.")
        self.assertIn("il-6", tokens)
        self.assertIn("p53", tokens)
        self.assertIn("hyphenated-terms", tokens)
        self.assertNotIn("and", tokens)
        self.assertEqual(tokenize_text("studies studied", stem=False), ["studies", "studied"])
        self.assertEqual(BM25Settings(fields=("title",), depth=100).fields, ("title",))

    def test_bm25_document_mapping_and_zero_scores(self) -> None:
        corpus = load_corpus(ROOT / "data/corpus.jsonl")
        indexes = [2, 0, 1]
        ranked = rank_document_indices(indexes, [0.0, 0.0, 0.0], corpus)
        self.assertEqual([doc_id for doc_id, _ in ranked], [
            corpus.records[0].doc_id,
            corpus.records[1].doc_id,
            corpus.records[2].doc_id,
        ])
        self.assertEqual([score for _, score in ranked], [0.0, 0.0, 0.0])
        deduplicated = rank_document_indices([1, 1, 0], [2.0, 1.0, 1.0], corpus)
        self.assertEqual([doc_id for doc_id, _ in deduplicated], [
            corpus.records[1].doc_id,
            corpus.records[0].doc_id,
        ])

    def test_bm25_toy_query_ranks_matching_document_first(self) -> None:
        import scripts.retrieve as retrieve_module

        records = [
            type("Record", (), {"doc_id": index, "title": text, "abstract": []})()
            for index, text in enumerate(
                ["cats eat fish", "dogs chase cats", "fish swim in water", "birds fly high"]
            )
        ]
        corpus = Corpus(records, {record.doc_id: index for index, record in enumerate(records)})

        class FakeBM25:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def index(self, tokens: object, **_kwargs: object) -> None:
                pass

            def retrieve(self, query_tokens: list[list[str]], **_kwargs: object) -> tuple[np.ndarray, np.ndarray]:
                query = set(query_tokens[0])
                scores = np.array([
                    len(query.intersection(text.split()))
                    for text in ["cats eat fish", "dogs chase cats", "fish swim in water", "birds fly high"]
                ], dtype=float)
                order = np.argsort(-scores, kind="stable")
                return order[None, :], scores[order][None, :]

        fake_bm25s = SimpleNamespace(
            BM25=FakeBM25,
            tokenize=lambda texts, **_kwargs: [text.lower().split() for text in texts],
        )
        with patch.object(retrieve_module.importlib, "import_module", return_value=fake_bm25s):
            ranking = BM25Retriever(corpus, BM25Settings(depth=100)).rank_claim("cats fish")
        self.assertEqual(ranking[0][0], 0)

    def test_dense_normalization_prefix_and_parameter_limit(self) -> None:
        normalized = normalize_embeddings(np.array([[3.0, 4.0], [1.0, 0.0]]))
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])
        bge = DenseSettings(name="bge", query_prefix="Represent: ")
        scincl = DenseSettings(name="scincl")
        self.assertEqual(apply_query_prefix("claim", bge), "Represent: claim")
        self.assertEqual(apply_query_prefix("claim", scincl), "claim")

        class Parameter:
            def __init__(self, size: int) -> None:
                self.size = size

            def numel(self) -> int:
                return self.size

        class Model:
            def parameters(self) -> list[Parameter]:
                return [Parameter(400_000_001)]

        with self.assertRaisesRegex(ValueError, "exceeding"):
            count_model_parameters(Model())

    def test_dense_mapping_and_deterministic_ties(self) -> None:
        loaded_corpus = load_corpus(ROOT / "data/corpus.jsonl")
        records = loaded_corpus.records[:3]
        corpus = Corpus(records, {record.doc_id: index for index, record in enumerate(records)})
        documents = np.zeros((3, 2), dtype=np.float32)
        documents[:, 0] = 1.0
        ranked = rank_embeddings(np.array([1.0, 0.0]), documents, corpus, 100)
        self.assertEqual([doc_id for doc_id, _ in ranked], [
            corpus.records[0].doc_id,
            corpus.records[1].doc_id,
            corpus.records[2].doc_id,
        ])

    def test_timing_statistics(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertEqual(
            latency_statistics([1.0, 2.0, 3.0]),
            {"mean_query_latency_ms": 2.0, "p95_query_latency_ms": 2.9},
        )

    def test_finetune_training_pairs_and_batches(self) -> None:
        corpus = load_corpus(ROOT / "data/corpus.jsonl")
        claims = load_claims(ROOT / "data/train_claims.jsonl")
        pairs = build_positive_pairs(claims, corpus)
        self.assertTrue(pairs)
        self.assertNotIn(0, {pair.claim_id for pair in pairs})
        batches = batch_without_duplicates(pairs[:20], 4)
        for batch in batches:
            self.assertEqual(len({pair.claim_id for pair in batch}), len(batch))
            self.assertEqual(len({pair.doc_id for pair in batch}), len(batch))

    def test_finetune_hard_negatives_exclude_gold(self) -> None:
        corpus = load_corpus(ROOT / "data/corpus.jsonl")
        claims = load_claims(ROOT / "data/train_claims.jsonl")
        record = next(item for item in claims.records if item.evidence)
        positive = record.evidence[0].doc_id
        alternatives = [
            item.doc_id for item in corpus.records if item.doc_id != positive
        ][:3]
        bm25 = {record.claim_id: [(positive, 1.0), (alternatives[0], 0.9), (alternatives[1], 0.8)]}
        dense = {record.claim_id: [(alternatives[2], 1.0), (positive, 0.9)]}
        examples = mine_hard_negatives(
            type(claims)([record], {record.claim_id: 0}),
            corpus,
            bm25,
            dense,
            min_rank=2,
            max_rank=3,
        )
        self.assertTrue(examples)
        self.assertTrue(all(example.negative_doc_id != positive for example in examples))
        self.assertTrue(all(example.negative_doc_id not in {
            item.doc_id for item in record.evidence
        } for example in examples))

    def test_finetune_configuration_and_checkpoint_selection(self) -> None:
        first = FineTuneSettings(base_model="base", output_dir="one", learning_rate=1e-5)
        second = FineTuneSettings(
            base_model=first.base_model,
            output_dir="two",
            learning_rate=first.learning_rate,
        )
        self.assertEqual(first.base_model, second.base_model)
        self.assertEqual(select_best_checkpoint({1: 0.3, 2: 0.8, 3: 0.8}), 2)
        with patch("scripts.finetune.importlib.import_module") as importer:
            losses = SimpleNamespace(MultipleNegativesRankingLoss=object())
            importer.return_value = losses
            self.assertIs(finetune_module._require_losses_api(), losses)

    def test_task4_loads_task1_task2_training_artifacts(self) -> None:
        corpus = load_corpus(ROOT / "data/corpus.jsonl")
        claims = load_claims(ROOT / "data/train_claims.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text(
                json.dumps({
                    str(claims.records[0].claim_id): [
                        [corpus.records[0].doc_id, 1.0]
                    ]
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _load_training_run(path, claims, corpus)

    def test_verification_labels_thresholds_and_metrics(self) -> None:
        class Config:
            id2label = {0: "neutral", 1: "entailment", 2: "contradiction"}

        self.assertEqual(read_label_mapping(Config()), {
            "neutral": 0, "entailment": 1, "contradiction": 2,
        })
        predictions = [(1, 10, "SUPPORT")]
        self.assertEqual(verification_metrics(predictions, predictions)["f1"], 1.0)
        selected, results = select_thresholds(
            [(0.5, 0.5, 10), (0.6, 0.6, 20)],
            lambda support, contradict, depth: {
                "precision": support, "recall": contradict, "f1": depth / 20,
            },
        )
        self.assertEqual(selected, (0.6, 0.6, 20))
        self.assertEqual(len(results), 2)

    def test_verification_outputs_one_label_per_claim_document(self) -> None:
        claims = Claims([ClaimRecord(7, "claim", [])], {7: 0})
        backend = object.__new__(NLIBackend)
        backend.label_indices = {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 2,
        }
        backend.settings = VerificationSettings(
            support_threshold=0.5,
            contradict_threshold=0.5,
            gate=0.1,
            candidate_depth=3,
        )
        cached = {
            7: [
                {"doc_id": 10, "probabilities": [0.7, 0.8, 0.1]},
                {"doc_id": 10, "probabilities": [0.6, 0.7, 0.1]},
                {"doc_id": 11, "probabilities": [0.8, 0.1, 0.1]},
            ],
        }

        predictions = backend.from_cache(claims, cached)

        self.assertEqual(predictions, [
            (7, 10, "SUPPORT"),
            (7, 11, "CONTRADICT"),
        ])
        self.assertIsNone(_best_verification_label(12, 0.4, 0.3, 0.5, 0.5))

    def test_q1_claim_selection_always_returns_two_per_direction(self) -> None:
        bm25_ids, dense_ids = _select_q1_claims({
            1: 0.5,
            2: -0.2,
            3: 0.1,
            4: -0.8,
        })
        self.assertEqual(bm25_ids, [1, 3])
        self.assertEqual(dense_ids, [4, 2])
        with self.assertRaisesRegex(ValueError, "at least two"):
            _select_q1_claims({1: 0.1})

    def test_report_normalizes_existing_verification_csv_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t5_verification.csv"
            path.write_text(
                "claim_id,doc_id,label\n"
                "1,10,SUPPORT\n"
                "1,10,SUPPORT\n"
                "1,11,CONTRADICT\n"
                "1,11,SUPPORT\n",
                encoding="utf-8",
            )

            predictions, conflicts = _normalize_verification_csv(path)

            self.assertEqual(predictions, [(1, 10, "SUPPORT")])
            self.assertEqual(conflicts, 1)
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ["claim_id,doc_id,label", "1,10,SUPPORT"],
            )

    def test_finetuned_nli_threshold_ties_do_not_prefer_empty_predictions(self) -> None:
        claims = Claims([ClaimRecord(1, "claim", [])], {1: 0})
        backend = object.__new__(NLIBackend)
        backend.label_indices = {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 2,
        }
        cached = {
            1: [{"doc_id": 10, "probabilities": [0.01, 0.2, 0.79]}],
        }
        settings = [
            (0.1, 0.1, 0.0, 3),
            (0.5, 0.5, 0.6, 1),
        ]

        def evaluate(support, contradict, gate, depth):
            backend.settings = VerificationSettings(
                support_threshold=support,
                contradict_threshold=contradict,
                gate=gate,
                candidate_depth=depth,
            )
            predictions = backend.from_cache(claims, cached)
            metrics = verification_metrics(predictions, {(2, 20, "SUPPORT")})
            metrics["prediction_count"] = float(len(predictions))
            return metrics

        selected, _ = select_thresholds(
            settings, evaluate, prefer_coverage_on_ties=True
        )
        self.assertEqual(selected, settings[0])
        evaluate(*selected)
        self.assertEqual(len(backend.from_cache(claims, cached)), 1)

        baseline_selected, _ = select_thresholds(settings, evaluate)
        self.assertEqual(baseline_selected, settings[1])
        evaluate(*baseline_selected)
        self.assertEqual(len(backend.from_cache(claims, cached)), 0)

    def test_verification_threshold_grid_uses_configured_values(self) -> None:
        grid = verification_threshold_grid({
            "support_thresholds": [0.25, 0.75],
            "contradict_thresholds": [0.4],
            "gate_thresholds": [0.6],
            "candidate_depths": [2, 4],
        })
        self.assertEqual(len(grid), 4)
        self.assertEqual(grid[0], (0.25, 0.4, 0.6, 2))
        self.assertEqual(grid[-1], (0.75, 0.4, 0.6, 4))

    def test_default_verification_config_uses_configured_grid(self) -> None:
        config = load_config(ROOT / "code" / "scripts" / "config.yaml")
        verification = config["verification"]
        grid = verification_threshold_grid(verification)
        expected_size = (
            len(verification["support_thresholds"])
            * len(verification["contradict_thresholds"])
            * len(verification["gate_thresholds"])
            * len(verification["candidate_depths"])
        )
        self.assertEqual(len(grid), expected_size)
        self.assertEqual(grid[0], (
            verification["support_thresholds"][0],
            verification["contradict_thresholds"][0],
            verification["gate_thresholds"][0],
            verification["candidate_depths"][0],
        ))
        self.assertEqual(grid[-1], (
            verification["support_thresholds"][-1],
            verification["contradict_thresholds"][-1],
            verification["gate_thresholds"][-1],
            verification["candidate_depths"][-1],
        ))

    def test_verification_score_cache_invalidates_on_identity_change(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            def predict(self, _pairs, **_kwargs):
                self.calls += 1
                return [[0.1, 0.8, 0.1]]

        backend = object.__new__(NLIBackend)
        backend.cache_model_identity = {"model_name": "test-model"}
        backend.settings = VerificationSettings(candidate_depth=1)
        backend.model = FakeModel()
        claims = Claims([ClaimRecord(1, "claim", [])], {1: 0})
        corpus = Corpus(
            [CorpusRecord(2, "title", ["sentence"])],
            {2: 0},
        )
        ranking = {1: [(2, 1.0)]}

        def premise_builder(_claim, _document):
            return "premise"

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "scores.json"
            backend.score_cache(
                claims, ranking, corpus, cache_path, premise_builder,
                cache_key="selector-v1",
            )
            backend.score_cache(
                claims, ranking, corpus, cache_path, premise_builder,
                cache_key="selector-v1",
            )
            self.assertEqual(backend.model.calls, 1)
            backend.score_cache(
                claims, ranking, corpus, cache_path, premise_builder,
                cache_key="selector-v2",
            )
            self.assertEqual(backend.model.calls, 2)
            backend.cache_model_identity = {
                "model_name": "test-model",
                "checkpoint": {"mtime_ns": 2},
            }
            backend.score_cache(
                claims, ranking, corpus, cache_path, premise_builder,
                cache_key="selector-v2",
            )
            self.assertEqual(backend.model.calls, 3)

    def test_verification_limit_csv_and_negation(self) -> None:
        class Parameter:
            def numel(self) -> int:
                return 400_000_001

        class Model:
            def parameters(self) -> list[Parameter]:
                return [Parameter()]

        with self.assertRaisesRegex(ValueError, "exceeding"):
            count_nli_parameters(Model())
        self.assertTrue(has_negation_cue("The treatment does not help."))
        self.assertFalse(has_negation_cue("notable results"))
        self.assertEqual(len(ALLOWED_CATEGORIES), 6)
        self.assertEqual(
            len(limit_predictions([(1, index, "SUPPORT") for index in range(5)])),
            3,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verification.csv"
            write_verification_csv(path, [(1, 2, "SUPPORT"), (1, 3, "CONTRADICT")])
            self.assertEqual(path.read_text(encoding="utf-8").splitlines()[0],
                             "claim_id,doc_id,label")

    def test_pipeline_integration_contract(self) -> None:
        config = load_config(ROOT / "code" / "scripts" / "config.yaml")
        for section in ("paths", "evaluation", "bm25", "dense", "fusion", "rerank", "finetune", "verification"):
            self.assertIn(section, config)
        self.assertIn("retrieval_runs", inspect.signature(run_task3).parameters)
        self.assertEqual(list(inspect.signature(run_pipeline).parameters), ["config_path"])

    def test_fusion_methods_and_missing_documents(self) -> None:
        left = {1: [(10, 100.0), (20, 1.0)]}
        right = {1: [(20, 1.0), (30, 0.5)]}
        fused = rrf_fuse([left, right], constant=1, depth=100)
        self.assertEqual([doc_id for doc_id, _ in fused[1]], [20, 10, 30])
        interpolated = interpolate_fuse(
            {1: [(10, 0.01), (20, 0.0)]},
            {1: [(20, 100.0), (30, 50.0)]},
            alpha=0.5,
        )
        self.assertEqual([doc_id for doc_id, _ in interpolated[1]][:3], [10, 20, 30])

    def test_hybrid_search_includes_weighted_rrf_baseline(self) -> None:
        bm25 = {1: [(10, 2.0), (20, 1.0)]}
        dense = {1: [(20, 0.9), (10, 0.8)]}
        gold = {1: [20]}
        score, spec = search_hybrid(
            bm25,
            dense,
            gold,
            rrf_constants=(10, 30, 60),
            dense_weights=(1.0, 1.5, 2.0, 3.0),
            interpolation_alphas=(0.5, 0.6, 0.7, 0.8, 0.9),
        )
        selected = apply_hybrid(spec, bm25, dense)
        selected_score = evaluate_retrieval(
            {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in selected.items()},
            gold,
        )["mean"]["ndcg@10"]
        baseline = rrf_fuse([bm25, dense], weights=(1.0, 1.0), constant=10)
        baseline_score = evaluate_retrieval(
            {claim_id: [doc_id for doc_id, _ in ranking] for claim_id, ranking in baseline.items()},
            gold,
        )["mean"]["ndcg@10"]
        self.assertEqual(score, selected_score)
        self.assertGreaterEqual(score, baseline_score)

    def test_hybrid_search_uses_supplied_candidate_values(self) -> None:
        import scripts.fuse as fuse_module

        bm25 = {1: [(10, 2.0), (20, 1.0)]}
        dense = {1: [(20, 0.9), (10, 0.8)]}
        candidates: list[dict[str, object]] = []

        def record_candidate(spec, *_args, **_kwargs):
            candidates.append(dict(spec))
            return bm25

        with patch.object(fuse_module, "apply_hybrid", side_effect=record_candidate):
            search_hybrid(
                bm25,
                dense,
                {1: [10]},
                rrf_constants=(17,),
                dense_weights=(2.5,),
                interpolation_alphas=(0.3,),
            )
        self.assertEqual(candidates, [
            {"method": "rrf", "rrf_constant": 17, "weights": [1.0, 2.5]},
            {"method": "interpolation", "alpha": 0.3},
        ])

    def test_nli_training_adds_neutral_examples_for_no_evidence_claims(self) -> None:
        corpus = load_corpus(ROOT / "data/corpus.jsonl")
        claims = load_claims(ROOT / "data/train_claims.jsonl")
        record = next(item for item in claims.records if not item.evidence)
        single_claim = type(claims)([record], {record.claim_id: 0})
        selector, nli = _training_examples(
            single_claim,
            corpus,
            {record.claim_id: [(corpus.records[0].doc_id, 1.0)]},
            lambda _claim, sentences, count: list(sentences[:count]),
            lambda _claim, document: "selected: " + " ".join(document.abstract[:3]),
            seed=42,
            nli_labels={"neutral": 1, "entailment": 0, "contradiction": 2},
        )
        self.assertEqual(selector, [])
        self.assertTrue(nli)
        self.assertTrue(all(label == 1 for _, _, label in nli))

    def test_dense_premise_selection_is_deterministic_and_restores_document_order(self) -> None:
        self.assertEqual(_select_premise_indices([0.8, 0.9, 0.9], 2), [1, 2])
        self.assertEqual(_select_premise_indices([0.9, 0.8, 0.7], 2), [0, 1])

    def test_nli_training_uses_inference_premise_and_mixes_gold_rationale(self) -> None:
        corpus = load_corpus(ROOT / "data/corpus.jsonl")
        claims = load_claims(ROOT / "data/train_claims.jsonl")
        record = next(item for item in claims.records if item.evidence)
        single_claim = type(claims)([record], {record.claim_id: 0})
        evidence = record.evidence[0]
        negative_doc = next(
            item for item in corpus.records if item.doc_id != evidence.doc_id
        )
        label_indices = {"neutral": 1, "entailment": 0, "contradiction": 2}
        selected_premise = f"selected document {evidence.doc_id}"
        rationale = " ".join(
            corpus.records[corpus.id_to_index[evidence.doc_id]].abstract[index]
            for index in evidence.sentences
        )
        _, examples = _training_examples(
            single_claim,
            corpus,
            {
                record.claim_id: [
                    (evidence.doc_id, 1.0),
                    (negative_doc.doc_id, 0.5),
                ]
            },
            lambda _claim, sentences, count: list(sentences[:count]),
            lambda _claim, document: (
                selected_premise
                if document.doc_id == evidence.doc_id
                else f"selected document {document.doc_id}"
            ),
            seed=42,
            nli_labels=label_indices,
        )

        gold_label = label_indices[
            "entailment" if evidence.label == "SUPPORT" else "contradiction"
        ]
        self.assertIn((selected_premise, record.claim, gold_label), examples)
        self.assertIn((rationale, record.claim, gold_label), examples)
        self.assertIn(
            (f"selected document {negative_doc.doc_id}", record.claim, label_indices["neutral"]),
            examples,
        )

        _, without_rationale = _training_examples(
            single_claim,
            corpus,
            {
                record.claim_id: [
                    (evidence.doc_id, 1.0),
                    (negative_doc.doc_id, 0.5),
                ]
            },
            lambda _claim, sentences, count: list(sentences[:count]),
            lambda _claim, document: (
                selected_premise
                if document.doc_id == evidence.doc_id
                else f"selected document {document.doc_id}"
            ),
            seed=42,
            nli_labels=label_indices,
            include_gold_rationales=False,
        )
        self.assertNotIn((rationale, record.claim, gold_label), without_rationale)

    def test_finetuned_selector_premise_builder_matches_task5_order_and_count(self) -> None:
        class FakeModel:
            def eval(self) -> None:
                pass

        class FakeSelector:
            model = FakeModel()
            tokenizer = None

            def predict(self, pairs, **_kwargs):
                self.pairs = pairs
                return [0.1, 0.9, 0.8]

        selector = FakeSelector()
        build_premise = _selector_premise_builder(
            selector,
            {"verification": {"premise_sentences": 2, "premise_max_length": 256}},
        )
        document = CorpusRecord(5, "title", ["first sentence", "second sentence", "third sentence"])

        self.assertEqual(build_premise("claim", document), "second sentence third sentence")
        self.assertEqual(selector.pairs, [
            ("claim", "first sentence"),
            ("claim", "second sentence"),
            ("claim", "third sentence"),
        ])

    def test_fine_tuned_nli_validation_comparison_uses_whole_abstract_baseline(self) -> None:
        comparison = compare_nli_validation_f1({
            "zero_shot_whole_abstract": 0.4,
            "fine_tuned_nli": 0.55,
        })
        self.assertEqual(comparison["whole_abstract_baseline_f1"], 0.4)
        self.assertEqual(comparison["fine_tuned_nli_f1"], 0.55)
        self.assertAlmostEqual(comparison["f1_delta"], 0.15)
        self.assertIsNone(compare_nli_validation_f1({
            "zero_shot_whole_abstract": 0.4,
        }))

    def test_rerank_order_and_parameter_limit(self) -> None:
        corpus = load_corpus(ROOT / "data/corpus.jsonl")
        ranking = [(corpus.records[i].doc_id, 0.0) for i in range(3)]
        def fake_scores(*args):
            if not args:
                raise AssertionError("pairs are required")
            return [0.1, 0.9]

        reranked, _ = rerank_one(
            "claim",
            ranking + [ranking[1]],
            corpus,
            fake_scores,
            candidate_k=2,
        )
        self.assertEqual([doc_id for doc_id, _ in reranked], [
            corpus.records[1].doc_id,
            corpus.records[0].doc_id,
            corpus.records[2].doc_id,
        ])

        class Parameter:
            def numel(self) -> int:
                return 400_000_001

        class Model:
            def parameters(self) -> list[Parameter]:
                return [Parameter()]

        with self.assertRaisesRegex(ValueError, "exceeding"):
            count_cross_encoder_parameters(Model())

    def test_rerank_candidate_selection_uses_ndcg(self) -> None:
        corpus = load_corpus(ROOT / "data/corpus.jsonl")
        claims = load_claims(ROOT / "data/val_claims.jsonl")
        record = next(item for item in claims.records if item.evidence)
        claims = type(claims)([record], {record.claim_id: 0})
        claim_id = record.claim_id
        gold_doc = record.evidence[0].doc_id
        runs = {claim_id: [(gold_doc, 1.0)]}

        class FakeReranker:
            def rerank(self, *args, candidate_k):
                if len(args) != 3:
                    raise AssertionError("rerank arguments are required")
                return (
                    {claim_id: [(gold_doc, 1.0)] if candidate_k == 10 else [(999, 1.0)]},
                    {},
                )

        self.assertEqual(
            select_candidate_k({"selected": runs}, claims, corpus, FakeReranker(), [10, 20])[0],
            10,
        )


if __name__ == "__main__":
    unittest.main()
