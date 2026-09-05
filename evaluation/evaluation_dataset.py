"""Evaluation schemas and RAGAS conversion helpers."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Type, TypeVar


@dataclass
class TestsetSample:
    user_input: str
    reference_contexts: List[str]
    ground_truth_answer: str

    def __post_init__(self) -> None:
        if not self.user_input.strip():
            raise ValueError("user_input must not be empty")
        if not self.reference_contexts or not all(
            isinstance(item, str) and item.strip() for item in self.reference_contexts
        ):
            raise ValueError("reference_contexts must be a non-empty list of text chunks")
        if not self.ground_truth_answer.strip():
            raise ValueError("ground_truth_answer must not be empty")


@dataclass
class RagasEvalSample(TestsetSample):
    response: str
    retrieved_contexts: List[str]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.response, str):
            raise ValueError("response must be a string")
        if not isinstance(self.retrieved_contexts, list) or not all(
            isinstance(item, str) for item in self.retrieved_contexts
        ):
            raise ValueError("retrieved_contexts must be a list of strings")


SampleT = TypeVar("SampleT", TestsetSample, RagasEvalSample)


def read_jsonl(path: str | Path, sample_type: Type[SampleT]) -> List[SampleT]:
    path = Path(path)
    if not path.exists():
        return []

    rows: List[SampleT] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                rows.append(sample_type(**payload))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL row {line_number} in {path}: {exc}") from exc
    return rows


def append_jsonl(path: str | Path, sample: TestsetSample | RagasEvalSample) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
        handle.flush()


def rewrite_jsonl(path: str | Path, samples: Iterable[TestsetSample | RagasEvalSample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    temp_path.replace(path)


def to_ragas_evaluation_dataset(samples: Iterable[RagasEvalSample]):
    """Convert local schemas to RAGAS SingleTurnSample/EvaluationDataset.

    ``ground_truth_answer`` remains part of our stored dataset, while the three
    requested metrics consume only the fields they declare.  We also map it to
    RAGAS ``reference`` for traceability, although none of the requested
    metrics relies on that answer field.
    """
    try:
        from ragas import EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
    except ImportError as exc:
        raise ImportError(
            "RAGAS is required for conversion. Install evaluation dependencies first."
        ) from exc

    ragas_samples = [
        SingleTurnSample(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
            reference_contexts=sample.reference_contexts,
            reference=sample.ground_truth_answer,
        )
        for sample in samples
    ]
    return EvaluationDataset(samples=ragas_samples)