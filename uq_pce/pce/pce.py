import numpy as np
import numpy.typing as npt
from scipy.linalg import lstsq

from .create_alphas import create_alphas
from ..model.model import get_dataset, INPUT_DISTRIBUTION, ModelType, NUM_VARIABLES


def compute_psi(
    nominalX: npt.NDArray[np.float64], P: int, idx_attr_map: dict[int, str]
) -> npt.NDArray[np.float64]:
    """
    Compute the polynomial chaos expansion basis functions for the given training data.

    Args
    ---------
    nominalX (np.ndarray): Training data samples.
    P (int): Degree of the polynomial.
    idx_attr_map (dict): Mapping from attribute names to indices.

    Returns
    ---------
    np.ndarray: Polynomial chaos expansion basis functions
    """
    # Create alphas containing at most degree P, with sum across dimensions <= P
    alphas = create_alphas(NUM_VARIABLES, P)

    # For all parameters, each N_train sample is used to evaluate the orthogonal polynomial from dimension 0 to P
    all_poly = np.full((NUM_VARIABLES, P + 1, nominalX.shape[0]), np.nan)
    for i in range(NUM_VARIABLES):
        param = idx_attr_map[i]
        distribution = INPUT_DISTRIBUTION[param]
        for k in range(P + 1):
            all_poly[i, k] = distribution.evaluate_polynomial(nominalX[:, i], k)

    num_alphas = alphas.shape[0]
    Psi = np.ones(
        (num_alphas, nominalX.shape[0])
    )  # defined as transpose to one in problem statement, this is more natural
    for alpha_idx in range(num_alphas):
        alpha = alphas[alpha_idx]
        for i in range(NUM_VARIABLES):
            Psi[alpha_idx] *= all_poly[i, alpha[i]]

    return Psi


def pce_get_coefficients(N_train: int, P: int) -> npt.NDArray[np.float64]:
    """
    Get the coefficients of the polynomial chaos expansion (PCE) for a given number of training samples and polynomial degree.

    Args
    ---------
    N_train (int): Number of training samples.
    P (int): Degree of the polynomial.

    Returns
    ---------
    np.ndarray: Coefficients of the polynomial chaos expansion.
    """
    # Generate training data
    Y_train, nominalX_train, idx_attr_map = get_dataset(N_train, ModelType.CALL)

    Psi = compute_psi(nominalX_train, P, idx_attr_map)

    coeff, _, _, _ = lstsq(Psi.T, Y_train, lapack_driver="gelsy")  # type: ignore
    assert isinstance(coeff, np.ndarray), "Coefficient is not a numpy array."
    assert coeff.shape[0] == Psi.shape[0], "Coefficient shape mismatch with Psi shape."
    return coeff
