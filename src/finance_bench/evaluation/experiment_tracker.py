from datetime import datetime
from pathlib import Path

from finance_bench.utils.io import save_json


class ExperimentTracker:
    def __init__(self):
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.run_dir = Path(
            f"outputs/runs/{timestamp}"
        )

        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_results(
        self,
        filename,
        data,
    ):
        save_json(
            data,
            self.run_dir / filename,
        )