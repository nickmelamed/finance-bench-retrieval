import numpy as np

from finance_bench.types.schemas import BootstrapConfig


class BootstrapCI:
    def __init__(
        self,
        config: BootstrapConfig
    ):
        self.n_bootstrap = config.n_bootstrap
        self.confidence_level = config.confidence_level
        self.seed = config.seed

    def compute(
        self,
        values,
    ):
        values = np.array(values)

        rng = np.random.default_rng(self.seed)

        means = []

        for _ in range(self.n_bootstrap):
            sample = rng.choice(
                values,
                size=len(values),
                replace=True,
            )

            means.append(np.mean(sample))

        lower = np.percentile(
            means,
            (1 - self.confidence_level) / 2 * 100,
        )

        upper = np.percentile(
            means,
            (1 + self.confidence_level) / 2 * 100,
        )

        return {
            "mean": float(np.mean(values)),
            "lower": float(lower),
            "upper": float(upper),
        }