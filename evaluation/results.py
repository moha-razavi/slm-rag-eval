"""Persist raw RAGAS scores and a compact summary report."""
import json
from datetime import datetime, timezone
from typing import Dict
from .config import (
    RAW_SCORES_JSONL_PATH,
    RESULTS_DIR,
    SUMMARY_JSON_PATH,
    SUMMARY_MD_PATH,
    ensure_directories,
)

METRIC_COLUMNS = (
    "non_llm_context_precision_with_reference",
    "non_llm_context_recall",
    "faithfulness_with_hhem",
)

def save_results(evaluation_result) -> Dict[str, float]:
    ensure_directories()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = evaluation_result.to_pandas()

    with RAW_SCORES_JSONL_PATH.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    summary: Dict[str, float] = {}
    for column in METRIC_COLUMNS:
        if column in frame.columns:
            summary[column] = float(frame[column].mean())

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_samples": int(len(frame)),
        "metrics": summary,
        "raw_scores_jsonl": str(RAW_SCORES_JSONL_PATH),
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# RAGAS Evaluation Summary",
        "",
        f"Samples: {len(frame)}",
        "",
        "| Metric | Mean score |",
        "|---|---:|",
    ]
    for metric, score in summary.items():
        lines.append(f"| `{metric}` | {score:.4f} |")
    lines.extend(
        [
            "",
            f"Raw per-sample scores: `{RAW_SCORES_JSONL_PATH}`",
            "",
            "Notes: the two context metrics are non-LLM string-similarity metrics. "
            "Faithfulness uses the external evaluator only for claim extraction and "
            "Vectara HHEM-2.1-Open locally for verification.",
        ]
    )
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary