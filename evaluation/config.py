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
'''
EVAL_API_KEY = os.getenv("RAG_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY")
EVAL_API_BASE_URL = os.getenv("RAG_EVAL_API_BASE_URL") or None  # for OpenAI-compatible APIs
EVAL_API_MODEL = os.getenv("RAG_EVAL_API_MODEL", "gpt-4o-mini")
EVAL_API_TEMPERATURE = float(os.getenv("RAG_EVAL_API_TEMPERATURE", "0"))
EVAL_API_TIMEOUT_SECONDS = float(os.getenv("RAG_EVAL_API_TIMEOUT_SECONDS", "120"))

TESTSET_SIZE = int(os.getenv("RAG_EVAL_TESTSET_SIZE", "20"))
TESTSET_RANDOM_SEED = int(os.getenv("RAG_EVAL_RANDOM_SEED", "42"))
MIN_REFERENCE_CHARS = int(os.getenv("RAG_EVAL_MIN_REFERENCE_CHARS", "180"))

RAG_TOP_K = int(os.getenv("RAG_EVAL_TOP_K", "3"))
HHEM_DEVICE = os.getenv("RAG_EVAL_HHEM_DEVICE", "cuda")  # or cuda if supported
HHEM_BATCH_SIZE = int(os.getenv("RAG_EVAL_HHEM_BATCH_SIZE", "10"))
'''
# Any OpenAI-compatible endpoint can be used by setting EVAL_API_BASE_URL.
# Set atleast:
EVAL_API_KEY = "YOUR_API_KEY"

# Optional environment variables:
EVAL_API_BASE_URL = "YOUAR_EVAL_API_BASE_URL" or None # for OpenAI-compatible APIs export example: "https://provider.example/v1"
EVAL_API_MODEL = "YOUAR_RAG_EVAL_API_MODEL"
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
