from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.dataset import EVALUATION_CASES
from memory_engine.engine import LocalMemoryEngine


def run_case(case: dict, case_number: int) -> bool:
    with TemporaryDirectory(
        prefix=f"local-memory-evaluation-{case_number}-"
    ) as temporary_directory:
        database_path = Path(temporary_directory) / "evaluation.db"
        engine = LocalMemoryEngine(
            database_path=str(database_path)
        )

        for message in case["messages"]:
            engine.ingest(message)

        results = engine.recall(
            query=case["query"],
            top_k=5,
            threshold=0.45,
        )

        actual_texts = [
            result["text"]
            for result in results
        ]

        expected_texts = case["expected"]

        if not expected_texts:
            passed = len(actual_texts) == 0
        else:
            passed = all(
                expected in actual_texts
                for expected in expected_texts
            )

        status = "PASS" if passed else "FAIL"

        print(
            f"{status:5} | "
            f"{case['name']}"
        )

        if not passed:
            print(f"       Expected: {expected_texts}")
            print(f"       Actual:   {actual_texts}")

    return passed


def main() -> None:
    passed_count = 0

    print("Evaluation results")
    print("------------------")

    for index, case in enumerate(
        EVALUATION_CASES,
        start=1,
    ):
        if run_case(case, index):
            passed_count += 1

    total = len(EVALUATION_CASES)
    accuracy = passed_count / total

    print()
    print(f"Passed: {passed_count}/{total}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
