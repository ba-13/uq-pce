from enum import Enum
import numpy as np
import numpy.typing as npt
from functools import partial
from scipy.stats import qmc

from .blackscholes import call_option_price, put_option_price
from .distributions import (
    Distribution,
    FixedDistribution,
    UniformDistribution,
    NormalDistribution,
)

seed = 42
np.random.seed(seed)

# Parameters taken from
# AAPL Mar-May 2025 Stock details, Jun 2027 Options details
# US Treasury 10Y
INPUT_DISTRIBUTION: dict[str, Distribution] = {
    "s": NormalDistribution(196, 0.6),
    "t": FixedDistribution(0),
    "T": FixedDistribution(2),
    "K": UniformDistribution(180, 215),
    "sigma": UniformDistribution(0.2, 0.8),
    "r": NormalDistribution(0.0427, 0.002),
}

# Number of non-fixed parameters
NUM_VARIABLES = 0
fixed_arguments = {}
for input_name, input_dist in INPUT_DISTRIBUTION.items():
    if isinstance(input_dist, FixedDistribution):
        fixed_arguments[input_name] = input_dist.value
    else:
        NUM_VARIABLES += 1

sampler = qmc.LatinHypercube(d=NUM_VARIABLES, rng=seed)

# first consume the fixed parameters
call_option_price_partial = partial(call_option_price, **fixed_arguments)
put_option_price_partial = partial(put_option_price, **fixed_arguments)


class ModelType(Enum):
    CALL = call_option_price_partial
    PUT = put_option_price_partial


def get_dataset(
    N: int, model: ModelType = ModelType.CALL
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[int, str]]:
    """Generates a dataset of call and put option prices

    Args
    -------
        N (int): number of samples to generate

    Returns
    -------
        npt.NDArray[np.float64] : output of model (N)
        npt.NDArray[np.float64] : nominal random variable samples (NUM_VARIABLES, N)
        dict[str, int] : mapping of input names to their indices in the sample array
    """
    # Generate samples from the Latin Hypercube sampler
    sample = sampler.random(N)
    idx = 0
    name_dimension_map = {}
    inputs = np.ones_like(sample)
    # Transform the samples to the desired distributions
    for input_name, input_dist in INPUT_DISTRIBUTION.items():
        input_dist: Distribution
        if isinstance(input_dist, FixedDistribution):
            continue
        name_dimension_map[input_name] = idx
        # convert to model space
        inputs[:, idx] = input_dist.isoprobabilistic(sample[:, idx])
        # convert to nominal space
        sample[:, idx] = input_dist.isoprobabilistic_nominal(sample[:, idx])
        idx += 1

    # Calculate call and put option prices
    outputs = model.value(
        **{name: inputs[:, idx] for name, idx in name_dimension_map.items()}
    )
    dimension_name_map = {i: name for name, i in name_dimension_map.items()}

    return outputs, sample, dimension_name_map
