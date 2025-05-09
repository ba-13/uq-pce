import numpy as np
import matplotlib.pyplot as plt
from uq_pce.pce.pce import compute_psi
from scipy.linalg import lstsq

from uq_pce.model.model import get_dataset, INPUT_DISTRIBUTION, ModelType, NUM_VARIABLES

N_train = 40
N_test = 1000
P = 2
Y_train, nominalX_train, idx_attr_map = get_dataset(N_train, ModelType.CALL)

# plot nominal random variables
# plt.figure()
# for i, param in idx_attr_map.items():
#     plt.subplot(2, 2, i + 1)
#     # plt.scatter(X_train[:, i], Y_train, alpha=0.5)
#     plt.hist(nominalX_train[:, i], bins=20, density=True)
#     plt.title(param)
# plt.show()

Psi = compute_psi(nominalX_train, 2, idx_attr_map)
coeff, _, _, _ = lstsq(Psi.T, Y_train, lapack_driver="gelsy")  # type: ignore
print(coeff)
Y_surrogate = Psi.T @ coeff

plt.figure(1, figsize=(8, 8))
plt.scatter(Y_train, Y_surrogate, alpha=0.5)
plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "r--")
plt.xlabel("True Option Prices (Y)")
plt.ylabel("Surrogate Prices")
plt.title("Training PCE vs Real")
plt.grid(True)

Y_test, nominalX_test, idx_attr_map = get_dataset(1000, ModelType.CALL)
Psi_test = compute_psi(nominalX_test, P, idx_attr_map)
Y_surrogate_test = Psi_test.T @ coeff

plt.figure(2, figsize=(8, 8))
plt.scatter(Y_test, Y_surrogate_test, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--")
plt.xlabel("True Option Prices (Y)")
plt.ylabel("Surrogate Prices")
plt.title("Test PCE vs Real")
plt.grid(True)
plt.show()
