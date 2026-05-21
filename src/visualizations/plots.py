from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


class PlotGenerator:
    def accuracy_plot(
        self,
        methods,
        accuracies,
        lower_bounds,
        upper_bounds,
    ):
        plt.figure(figsize=(8, 5))

        errors = [
            [a - l for a, l in zip(accuracies, lower_bounds)],
            [u - a for a, u in zip(accuracies, upper_bounds)],
        ]

        plt.bar(
            methods,
            accuracies,
            yerr=errors,
            capsize=5,
        )

        plt.ylabel("Accuracy")
        plt.title("QA Accuracy with 95% Bootstrap CI")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "accuracy_ci.png",
            dpi=300,
        )

    def pareto_plot(
        self,
        methods,
        accuracies,
        token_efficiencies,
    ):
        plt.figure(figsize=(8, 5))

        plt.scatter(
            token_efficiencies,
            accuracies,
        )

        for i, method in enumerate(methods):
            plt.annotate(
                method,
                (
                    token_efficiencies[i],
                    accuracies[i],
                ),
            )

        plt.xlabel("Tokens per Correct Answer")
        plt.ylabel("Accuracy")

        plt.title("Performance–Efficiency Frontier")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "pareto_frontier.png",
            dpi=300,
        )

    def failure_plot(
        self,
        failure_results,
    ):
        """
        failure_results:
        {
            "bm25": {
                "retrieval_miss": 10,
                "hallucination": 4,
            },
            ...
        }
        """

        methods = list(
            failure_results.keys()
        )

        all_failure_types = sorted(
            {
                failure_type
                for failures in failure_results.values()
                for failure_type in failures.keys()
            }
        )

        x = np.arange(
            len(all_failure_types)
        )

        width = 0.18

        plt.figure(figsize=(10, 5))

        for i, method in enumerate(methods):

            values = [
                failure_results[method].get(
                    ft,
                    0,
                )
                for ft in all_failure_types
            ]

            plt.bar(
                x + i * width,
                values,
                width,
                label=method,
            )

        plt.xticks(
            x + width * (
                len(methods) - 1
            ) / 2,
            all_failure_types,
            rotation=15,
        )

        plt.ylabel("Count")

        plt.title(
            "Failure Modes by Retriever"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "failure_modes.png",
            dpi=300,
        )