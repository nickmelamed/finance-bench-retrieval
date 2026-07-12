import numpy as np

from src.evaluation.bootstrap import BootstrapCI
from src.types.schemas import BootstrapConfig


def test_bootstrap_ci_contains_mean():
    np.random.seed(42)

    values = [1, 1, 0, 1, 1, 0, 1]

    bootstrap = BootstrapCI(
        BootstrapConfig(enabled=True, n_bootstrap=500, confidence_level=0.95)
    )

    result = bootstrap.compute(values)

    assert result["mean"] == sum(values) / len(values)
    assert result["lower"] <= result["mean"] <= result["upper"]


def test_bootstrap_ci_narrows_with_more_samples():
    np.random.seed(42)

    config = BootstrapConfig(enabled=True, n_bootstrap=500, confidence_level=0.95)

    small = BootstrapCI(config).compute([1, 0, 1, 0])
    large = BootstrapCI(config).compute([1, 0, 1, 0] * 25)

    small_width = small["upper"] - small["lower"]
    large_width = large["upper"] - large["lower"]

    assert large_width < small_width
