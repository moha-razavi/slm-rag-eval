# RAGAS Evaluation Summary

Samples: 20

| Metric | Mean score |
|---|---:|
| `non_llm_context_precision_with_reference` | 0.6000 |
| `non_llm_context_recall` | 0.6500 |
| `faithfulness_with_hhem` | 0.8792 |

Raw per-sample scores: `D:\Work\rag_ragas_eval\EvaluationData\results\raw_scores.csv`

Notes: the two context metrics are non-LLM string-similarity metrics. Faithfulness uses the external evaluator only for claim extraction and Vectara HHEM-2.1-Open locally for verification.
