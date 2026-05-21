import json
from pathlib import Path

from src.ingest.financebench_dataset import (
    FinanceBenchDataset,
)


OUTPUT_PATH = (
    "data/processed/"
    "financebench_examples.json"
)


def main():

    dataset = FinanceBenchDataset()

    examples = dataset.load()

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            examples,
            f,
            indent=2,
        )

    print(
        f"Saved {len(examples)} examples"
    )


if __name__ == "__main__":
    main()