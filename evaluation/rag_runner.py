"""Run the existing local RAG pipeline over the generated evaluation questions."""
from typing import List
from src.rag_pipeline import LocalRAGPipeline
from .config import RAGAS_DATASET_PATH, RAG_TOP_K, SOURCE_MARKDOWN_PATH, TESTSET_PATH, ensure_directories
from .evaluation_dataset import (
    RagasEvalSample,
    TestsetSample,
    append_jsonl,
    read_jsonl,
)


def run_rag_dataset() -> List[RagasEvalSample]:
    ensure_directories()
    testset = read_jsonl(TESTSET_PATH, TestsetSample)
    if not testset:
        raise FileNotFoundError(
            f"No testset found at {TESTSET_PATH}. Run datasetbuilder.build_testset() first."
        )

    completed = read_jsonl(RAGAS_DATASET_PATH, RagasEvalSample)
    completed_by_question = {sample.user_input: sample for sample in completed}

    missing = [sample for sample in testset if sample.user_input not in completed_by_question]
    if not missing:
        return [completed_by_question[sample.user_input] for sample in testset]

    # Heavy local models/index are initialized once, then reused for all rows.
    pipeline = LocalRAGPipeline(markdown_path=SOURCE_MARKDOWN_PATH, top_k=RAG_TOP_K)

    for index, sample in enumerate(missing, start=1):
        response, retrieved_contexts = pipeline.run(sample.user_input)
        completed_sample = RagasEvalSample(
            user_input=sample.user_input,
            reference_contexts=sample.reference_contexts,
            ground_truth_answer=sample.ground_truth_answer,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        # Append after every row so an interrupted run can resume safely.
        append_jsonl(RAGAS_DATASET_PATH, completed_sample)
        completed_by_question[sample.user_input] = completed_sample
        print(f"Completed RAG row {index}/{len(missing)}")

    return [completed_by_question[sample.user_input] for sample in testset]


if __name__ == "__main__":
    run_rag_dataset()