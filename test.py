import numpy as np
import matplotlib.pyplot as plt
from uq_pce.pce.pce import compute_psi
from scipy.linalg import lstsq

from uq_pce.model.model import get_dataset, INPUT_DISTRIBUTION, ModelType, NUM_VARIABLES
from uq_pce.model.utils import plot_input_vs_output

N_test = 10000
P = 3


# calculate rmse
def rmse(Y1, Y2):
    return np.sqrt(np.mean((Y1 - Y2) ** 2))


Y_train, Y_test, Y_surrogate, Y_surrogate_test = (None,) * 4


def train_validate_loop(N_train):
    global Y_train, Y_test, Y_surrogate, Y_surrogate_test
    Y_train, nominalX_train, idx_attr_map = get_dataset(N_train, ModelType.CALL)
    # plot_input_vs_output(nominalX_train, Y_train, idx_attr_map, 2)

    Psi = compute_psi(nominalX_train, P, idx_attr_map)
    coeff, _, _, _ = lstsq(Psi.T, Y_train, lapack_driver="gelsy")  # type: ignore
    Y_surrogate = Psi.T @ coeff

    train_rmse = rmse(Y_train, Y_surrogate)

    Y_test, nominalX_test, idx_attr_map = get_dataset(N_test, ModelType.CALL)
    Psi_test = compute_psi(nominalX_test, P, idx_attr_map)
    Y_surrogate_test = Psi_test.T @ coeff

    test_rmse = rmse(Y_test, Y_surrogate_test)
    return train_rmse, test_rmse


train_rmses = []
test_rmses = []
Ns = range(50, 200, 1)
for N in Ns:
    train_rmse, test_rmse = train_validate_loop(N)
    train_rmses.append(train_rmse)
    test_rmses.append(test_rmse)

# plot train_rmses and test_rmses with rmse in log scale
plt.figure(figsize=(8, 6))
plt.plot(Ns, train_rmses, label="Train RMSE")
plt.plot(Ns, test_rmses, label="Test RMSE")
plt.xlabel("N_train")
plt.ylabel("RMSE")
plt.title("RMSE vs N_train")
plt.yscale("log")
plt.legend()
plt.grid()

# 25 seems to be just above the meeting point of train-validate loss
train_validate_loop(130)
assert Y_train is not None
assert Y_surrogate is not None
assert Y_test is not None
assert Y_surrogate_test is not None
plt.figure(figsize=(8, 8))
plt.scatter(Y_train, Y_surrogate, alpha=0.5)
plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "r--")
plt.xlabel("True Option Prices (Y)")
plt.ylabel("Surrogate Prices")
plt.title("Training PCE vs Real")
plt.grid(True)

plt.figure(figsize=(8, 8))
plt.scatter(Y_test, Y_surrogate_test, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--")
plt.xlabel("True Option Prices (Y)")
plt.ylabel("Surrogate Prices")
plt.title("Test PCE vs Real")
plt.grid(True)

plt.figure()
plt.hist(Y_surrogate_test, bins=50)
plt.xlabel("Y values")
plt.ylabel("Count")
plt.show()
