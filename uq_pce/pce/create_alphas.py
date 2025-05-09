import numpy as np
import numpy.typing as npt


def create_alphas(n_inputs: int, P: int) -> npt.NDArray[np.int32]:
    """Inbuilt truncation of alpha indices for polynomial chaos expansion.

    Args
    -------
        n_inputs (int): Dimension of Input space.
        P (int): Degree uptil consideration

    Returns
    -------
        alphas: all partitions of P into n_inputs
    """
    return np.array(
        list(filter(lambda x: sum(x) <= P, np.ndindex((P + 1,) * n_inputs)))
    )
