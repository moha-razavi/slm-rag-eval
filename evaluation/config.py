"""Configuration for RAGAS evaluation and evaluation artifacts."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_PDF_PATH = DATA_DIR / "document.pdf"
SOURCE_MARKDOWN_PATH = DATA_DIR / "document.md"
EVALUATION_DATA_DIR = PROJECT_ROOT / "EvaluationData"
TESTSET_DIR = EVALUATION_DATA_DIR / "testset"
RAGAS_DATASET_DIR = EVALUATION_DATA_DIR / "ragas_dataset"
RESULTS_DIR = EVALUATION_DATA_DIR / "results"
TESTSET_PATH = TESTSET_DIR / "testset.jsonl"
RAGAS_DATASET_PATH = RAGAS_DATASET_DIR / "ragas_dataset.jsonl"
RAW_SCORES_PATH = RESULTS_DIR / "raw_scores.csv"
RAW_SCORES_JSONL_PATH = RESULTS_DIR / "raw_scores.jsonl"
SUMMARY_JSON_PATH = RESULTS_DIR / "summary.json"
SUMMARY_MD_PATH = RESULTS_DIR / "summary.md"

# Any OpenAI-compatible endpoint can be used by setting EVAL_API_BASE_URL.
# Set atleast:
EVAL_API_KEY = "YOUR_API_KEY"
EVAL_API_BASE_URL = None   # Fill it for OpenAI-compatible APIs. example: "https://provider.example/v1"
EVAL_API_MODEL = "YOUAR_RAG_EVAL_API_MODEL"

# Optional environment variables: (shown values are already the defaults)
EVAL_API_TEMPERATURE = 0.0
EVAL_API_TIMEOUT_SECONDS = 120.0

TESTSET_SIZE = 20
TESTSET_RANDOM_SEED = 42
MIN_REFERENCE_CHARS = 180
RAG_TOP_K = 3
HHEM_DEVICE = "cpu"
HHEM_BATCH_SIZE = 10

def ensure_directories() -> None:
    for path in (TESTSET_DIR, RAGAS_DATASET_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)