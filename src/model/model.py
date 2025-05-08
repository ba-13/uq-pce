from enum import Enum
import numpy as np
import numpy.typing as npt
from functools import partial
from black_scholes import call_option_price, put_option_price
from scipy.stats import qmc, norm, Normal, Uniform


class Distribution:
    def sample(self, N: int) -> npt.NDArray[np.float64]:
        raise NotImplementedError("This is an abstract class method")


class FixedDistribution(Distribution):
    def __init__(self, value: float):
        self.value = value

    def sample(self, N: int) -> npt.NDArray[np.float64]:
        return np.full(N, self.value)


class NormalDistribution(Distribution):
    def __init__(self, mean: float, stddev: float):
        self.mean = mean
        self.stddev = stddev
        self.sampler = Normal(mu=mean, sigma=stddev)

    def sample(self, N: int) -> npt.NDArray[np.float64]:
        return self.sampler.sample(N)

    def isoprobabilistic(self, numbers: npt.NDArray[np.float64]):
        return self.sampler.icdf(numbers)


class UniformDistribution(Distribution):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self.sampler = Uniform(a=low, b=high)

    def sample(self, N: int) -> npt.NDArray[np.float64]:
        return self.sampler.sample(N)
    
    def isoprobabilistic(self, numbers: npt.NDArray[np.float64]):
        return self.sampler.icdf(numbers)


# Parameters taken from
# AAPL Mar-May 2025
# US Treasury 10Y
seed = 42
INPUT_DISTRIBUTION = {
    "s": NormalDistribution(196, 0.6),
    "t": FixedDistribution(0),
    "T": FixedDistribution(23),
    "K": UniformDistribution(180, 215),
    "sigma": UniformDistribution(0.2, 0.8),
    "r": NormalDistribution(4.27, 0.2),
}

NUM_VARIABLES = 0
fixed_arguments = {}
for input_name, input_dist in INPUT_DISTRIBUTION.items():
    if isinstance(input_dist, FixedDistribution):
        fixed_arguments[input_name] = input_dist.value
    else:
        NUM_VARIABLES += 1

call_option_price_partial = partial(call_option_price, **fixed_arguments)

put_option_price_partial = partial(put_option_price, **fixed_arguments)

np.random.seed(seed)
sampler = qmc.LatinHypercube(d=NUM_VARIABLES, rng=seed)
