from __future__ import annotations

import json
from pathlib import Path

RUNS_DIR = Path("outputs/runs")

DASHBOARD_DIR = Path("dashboard")

OUTPUT_PATH = DASHBOARD_DIR / "data.js"


def load_runs() -> list[dict]:
    runs = []

    for run_dir in sorted(RUNS_DIR.glob("*")):
        results_path = run_dir / "evaluation_results.json"

        if not results_path.exists():
            continue

        with open(results_path, "r", encoding="utf-8") as f:
            methods = json.load(f)

        runs.append(
            {
                "run_id": run_dir.name,
                "methods": methods,
            }
        )

    return runs


def main():
    runs = load_runs()

    if not runs:
        raise SystemExit(
            f"No evaluation runs found under {RUNS_DIR}/ - "
            f"run `make evaluate` first."
        )

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(runs, indent=2)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"window.EVAL_RUNS = {payload};\n")

    print(f"Wrote {len(runs)} run(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
