# SciFact NLP Bootcamp — Validation Report

**Student:** Jyotirya Agrawal  
**Roll number:** 24B1300  
**Seed:** 42  
**Evaluation basis:** validation split only

## Q1. BM25 and dense retrieval differences

BM25 substantially outperformed the general dense retriever on claims **673** (nDCG@10: 1.000 vs. 0.356) and **381** (1.000 vs. 0.500). These claims likely benefit from distinctive biomedical terms and direct lexical overlap. Dense retrieval performed better on claims **766** (1.000 vs. 0.000) and **1311** (1.000 vs. 0.333), where semantic matching can help despite vocabulary or phrasing differences. This contrast illustrates the complementary strengths of lexical matching and semantic representations.

## Q2. Largest Task 3 validation gain and latency

Hybrid fusion provided the largest positive gain among the Task 3 combination/reranking components: validation nDCG@10 rose from **0.8976** for general dense retrieval to **0.9098**, a gain of **0.0121**. Its measured latency was approximately **0.248 ms/query**. The subsequent cross-encoder reranking result was **0.9028**, below the hybrid score, at approximately **595.7 ms/query**.

## Q3. Effect of hard negatives and overfitting

Hard-negative fine-tuning achieved validation nDCG@10 of **0.8954**, compared with **0.8907** for random negatives and **0.8648** for the base model. The hard-negative run therefore improved by about **0.0048** over random negatives. Across the three recorded epochs, validation nDCG increased while training loss decreased for both runs; there was no observed validation decline suggesting overfitting in these measurements.

## Q4. Retrieval misses among verification errors

There were **5 retrieval misses among 71 validation verification errors**, or approximately **7.0%**. For these cases, the annotated evidence document was absent from the retrieved candidates, so the verifier could not select it regardless of its NLI decision.

## Q5. Next change

I would next fine-tune and evaluate the NLI classifier using premises selected from the actual retrieved candidate documents, with the same sentence-selection path used at inference. The current validation result selected the zero-shot whole-abstract system (F1 **0.574**); the fine-tuned NLI variants did not produce useful predictions in the recorded run. Matching training examples to inference inputs is a focused way to address that gap before changing thresholds or the pretrained baseline.
