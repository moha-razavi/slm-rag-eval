"""
- NonLLMContextPrecisionWithReference: local/non-LLM
- NonLLMContextRecall: local/non-LLM
- FaithfulnesswithHHEM: external LLM only for claim extraction,
  local Vectara HHEM-2.1-Open for verification

This file is written for RAGAS 0.3.9 and uses LangchainLLMWrapper instead of
ragas.llms.llm_factory so the evaluator LLM exposes the async LangChain
prompt-generation interface expected by this RAGAS version.
"""
from typing import Iterable
from .config import (
    EVAL_API_BASE_URL,
    EVAL_API_KEY,
    EVAL_API_MODEL,
    EVAL_API_TEMPERATURE,
    EVAL_API_TIMEOUT_SECONDS,
    HHEM_BATCH_SIZE,
    HHEM_DEVICE,
)
from .evaluation_dataset import RagasEvalSample, to_ragas_evaluation_dataset

def _build_claim_extraction_llm():
    """Build the external OpenAI-compatible LLM used only for claim extraction."""
    if not EVAL_API_KEY:
        raise RuntimeError("EVAL_API_KEY is not configured in evaluation/config.py")

    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    kwargs = {
        "model": EVAL_API_MODEL,
        "api_key": EVAL_API_KEY,
        "temperature": EVAL_API_TEMPERATURE,
        "timeout": EVAL_API_TIMEOUT_SECONDS,
    }

    if EVAL_API_BASE_URL:
        kwargs["base_url"] = EVAL_API_BASE_URL

    langchain_llm = ChatOpenAI(**kwargs)
    return LangchainLLMWrapper(langchain_llm)


def run_metrics(samples: Iterable[RagasEvalSample]):
    """Evaluate all samples with the three required RAGAS metrics."""
    try:
        import ragas
        from ragas import evaluate
        from ragas.metrics import (
            FaithfulnesswithHHEM,
            NonLLMContextPrecisionWithReference,
            NonLLMContextRecall,
        )
    except ImportError as exc:
        raise ImportError(
            "RAGAS evaluation dependencies are missing or incompatible. "
            "This project currently expects ragas==0.3.9."
        ) from exc

    ragas_dataset = to_ragas_evaluation_dataset(list(samples))
    claim_llm = _build_claim_extraction_llm()

    metrics = [
        NonLLMContextPrecisionWithReference(),
        NonLLMContextRecall(),
        FaithfulnesswithHHEM(
            llm=claim_llm,
            device=HHEM_DEVICE,
            batch_size=HHEM_BATCH_SIZE,
        ),
    ]

    print(f"RAGAS version: {getattr(ragas, '__version__', 'unknown')}")

    return evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        raise_exceptions=True,
    )