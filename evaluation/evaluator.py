"""End-to-end evaluation orchestration."""
import argparse
from .datasetbuilder import build_testset
from .metrics import run_metrics
from .rag_runner import run_rag_dataset
from .results import save_results

def run_evaluation(testset_size: int | None = None, force_testset: bool = False):
    kwargs = {"force": force_testset}
    if testset_size is not None:
        kwargs["size"] = testset_size

    build_testset(**kwargs)
    rag_samples = run_rag_dataset()
    evaluation_result = run_metrics(rag_samples)
    summary = save_results(evaluation_result)
    return evaluation_result, summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate, run, and score the local RAG evaluation set.")
    parser.add_argument("--testset-size", type=int, default=None)
    parser.add_argument(
        "--force-testset",
        action="store_true",
        help="Delete and regenerate the synthetic testset (uses external API calls).",
    )
    args = parser.parse_args()

    _result, summary = run_evaluation(
        testset_size=args.testset_size,
        force_testset=args.force_testset,
    )
    print("Evaluation summary:")
    for metric, value in summary.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()