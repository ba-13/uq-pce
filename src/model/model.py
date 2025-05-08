from enum import Enum
from typing import Callable
import numpy as np
import numpy.typing as npt
from functools import partial
from blackscholes import call_option_price, put_option_price
from scipy.stats import qmc, norm, Normal, Uniform

seed = 42
np.random.seed(seed)


class Distribution:
    def sample(self, N: int) -> npt.NDArray[np.float32]:
        raise NotImplementedError("This is an abstract class method")

    def isoprobabilistic(self, numbers: npt.NDArray[np.float32]):
        raise NotImplementedError("This is an abstract class method")


class FixedDistribution(Distribution):
    def __init__(self, value: float):
        self.value = value

    def sample(self, N: int) -> npt.NDArray[np.float32]:
        return np.full(N, self.value)


class NormalDistribution(Distribution):
    def __init__(self, mean: float, stddev: float):
        self.mean = mean
        self.stddev = stddev
        self.sampler = Normal(mu=mean, sigma=stddev)

    def sample(self, N: int) -> npt.NDArray[np.float32]:
        return self.sampler.sample(N)

    def isoprobabilistic(self, numbers: npt.NDArray[np.float32]):
        return self.sampler.icdf(numbers)


class UniformDistribution(Distribution):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self.sampler = Uniform(a=low, b=high)

    def sample(self, N: int) -> npt.NDArray[np.float32]:
        return self.sampler.sample(N)

    def isoprobabilistic(self, numbers: npt.NDArray[np.float32]):
        return self.sampler.icdf(numbers)


# Parameters taken from
# AAPL Mar-May 2025
# US Treasury 10Y
INPUT_DISTRIBUTION = {
    "s": NormalDistribution(196, 0.6),
    "t": FixedDistribution(0),
    "T": FixedDistribution(2),
    "K": UniformDistribution(180, 215),
    "sigma": UniformDistribution(0.2, 0.8),
    "r": NormalDistribution(0.0427, 0.002),
}

NUM_VARIABLES = 0
fixed_arguments = {}
for input_name, input_dist in INPUT_DISTRIBUTION.items():
    if isinstance(input_dist, FixedDistribution):
        fixed_arguments[input_name] = input_dist.value
    else:
        NUM_VARIABLES += 1

sampler = qmc.LatinHypercube(d=NUM_VARIABLES, rng=seed)

call_option_price_partial = partial(call_option_price, **fixed_arguments)
put_option_price_partial = partial(put_option_price, **fixed_arguments)


def get_dataset(
    N: int, model: Callable = call_option_price_partial
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], dict[str, int]]:
    """Generates a dataset of call and put option prices

    Args:
        N (int): number of samples to generate

    Returns:
        npt.NDArray[np.float32]: dataset of call and put option prices
    """
    # Generate samples from the Latin Hypercube sampler
    sample = sampler.random(N)
    idx = 0
    name_dimension_map = {}
    # Transform the samples to the desired distributions
    for input_name, input_dist in INPUT_DISTRIBUTION.items():
        input_dist: Distribution
        if isinstance(input_dist, FixedDistribution):
            continue
        name_dimension_map[input_name] = idx
        sample[:, idx] = input_dist.isoprobabilistic(sample[:, idx])
        idx += 1

    # Calculate call and put option prices
    outputs = model(
        **{name: sample[:, idx] for name, idx in name_dimension_map.items()}
    )

    return outputs, sample, name_dimension_map
