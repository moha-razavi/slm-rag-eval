"""Generates the reference testset once using an external API.

External LLM calls are intentionally confined to this module.  Reference
contexts are never synthesized: they are exact chunk texts returned by the
existing ``src.chunking.chunk_text`` implementation.
"""
import json
import random
import re
from typing import Dict, List
from src.chunking import chunk_text
from .config import (
    EVAL_API_BASE_URL,
    EVAL_API_KEY,
    EVAL_API_MODEL,
    EVAL_API_TEMPERATURE,
    EVAL_API_TIMEOUT_SECONDS,
    MIN_REFERENCE_CHARS,
    SOURCE_MARKDOWN_PATH,
    TESTSET_PATH,
    TESTSET_RANDOM_SEED,
    TESTSET_SIZE,
    ensure_directories,
)
from .evaluation_dataset import TestsetSample, append_jsonl, read_jsonl


SYSTEM_PROMPT = """You create high-quality question-answer examples for evaluating a RAG system.
Use ONLY the supplied reference chunk. Return one natural question that is answerable from that chunk and a concise factual answer.
Rules:
- The question must require information present in the chunk.
- Do not mention 'the chunk', 'the context', page numbers, or citations.
- Avoid yes/no questions and vague prompts.
- Do not use outside knowledge.
- Output exactly one JSON object with keys: user_input, ground_truth_answer.
"""


def _client():
    if not EVAL_API_KEY:
        raise RuntimeError(
            "Missing evaluator API key. Set RAG_EVAL_API_KEY (or OPENAI_API_KEY)."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the 'openai' package to build the evaluation testset.") from exc

    kwargs = {"api_key": EVAL_API_KEY, "timeout": EVAL_API_TIMEOUT_SECONDS}
    if EVAL_API_BASE_URL:
        kwargs["base_url"] = EVAL_API_BASE_URL
    return OpenAI(**kwargs)


def _parse_json_object(text: str) -> Dict[str, str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"API did not return a JSON object: {text[:300]!r}")
        payload = json.loads(match.group(0))

    question = str(payload.get("user_input", "")).strip()
    answer = str(payload.get("ground_truth_answer", "")).strip()
    if not question or not answer:
        raise ValueError("API JSON must contain non-empty user_input and ground_truth_answer")
    return {"user_input": question, "ground_truth_answer": answer}


def _generate_example(client, reference_text: str) -> TestsetSample:
    response = client.chat.completions.create(
        model=EVAL_API_MODEL,
        temperature=EVAL_API_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Reference chunk:\n---\n{reference_text}\n---",
            },
        ],
    )
    content = response.choices[0].message.content or ""
    generated = _parse_json_object(content)
    return TestsetSample(
        user_input=generated["user_input"],
        reference_contexts=[reference_text],
        ground_truth_answer=generated["ground_truth_answer"],
    )


def _candidate_chunks() -> List[str]:
    chunks = chunk_text(str(SOURCE_MARKDOWN_PATH))
    candidates = [
        chunk["text"].strip()
        for chunk in chunks
        if len(chunk.get("text", "").strip()) >= MIN_REFERENCE_CHARS
    ]
    if not candidates:
        raise ValueError("No chunks are long enough to generate evaluation questions.")

    rng = random.Random(TESTSET_RANDOM_SEED)
    rng.shuffle(candidates)
    return candidates


def build_testset(size: int = TESTSET_SIZE, force: bool = False) -> List[TestsetSample]:
    """Build or resume the synthetic testset.

    If a complete testset already exists it is reused, so generation happens
    only once unless ``force=True`` is explicitly supplied.
    """
    ensure_directories()

    if force and TESTSET_PATH.exists():
        TESTSET_PATH.unlink()

    existing = read_jsonl(TESTSET_PATH, TestsetSample)
    if len(existing) >= size:
        return existing[:size]

    candidates = _candidate_chunks()
    if size > len(candidates):
        raise ValueError(f"Requested {size} samples but only {len(candidates)} candidate chunks exist.")

    client = _client()
    used_reference_texts = {sample.reference_contexts[0] for sample in existing}
    questions = {sample.user_input.casefold() for sample in existing}

    for reference_text in candidates:
        if len(existing) >= size:
            break
        if reference_text in used_reference_texts:
            continue

        sample = _generate_example(client, reference_text)
        if sample.user_input.casefold() in questions:
            continue

        append_jsonl(TESTSET_PATH, sample)
        existing.append(sample)
        used_reference_texts.add(reference_text)
        questions.add(sample.user_input.casefold())
        print(f"Generated test sample {len(existing)}/{size}")

    if len(existing) < size:
        raise RuntimeError(f"Only generated {len(existing)} of {size} requested test samples.")
    return existing


if __name__ == "__main__":
    build_testset()