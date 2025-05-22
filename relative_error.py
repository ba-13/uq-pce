import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from uq_pce.model.model import get_dataset, ModelType, NUM_VARIABLES
from uq_pce.pce.pce import compute_psi, create_alphas

np.random.seed(40)
N = 10000
p = 3  # degree of truncation

# calculate the original model mean and variance using Monte-Carlo
Y, nominalX, idx_attr_map = get_dataset(N, ModelType.CALL)
original_mean = Y.mean(axis=0)
original_var = Y.var(axis=0)
# print(f"mu={model_mean}, var={model_var}")

# 2.6.3: Fixed n, increasing p
p = 1
N = 80
Y, nominalX, idx_attr_map = get_dataset(N, ModelType.CALL)
ps = []
Ps = []
rerror_means = []
rerror_vars = []
while True:
    alphas = create_alphas(NUM_VARIABLES, p)
    P = alphas.shape[0]
    if P > N:
        break
    Psi = compute_psi(nominalX, p, idx_attr_map)
    # Solve least-squares problem to obtain coefficients
    coeff, _, _, _ = lstsq(Psi.T, Y, lapack_driver="gelsy")  # type: ignore
    Y_surrogate = Psi.T @ coeff
    ps.append(p)
    Ps.append(P)
    model_mean = coeff[0]
    model_var = np.sum(coeff[1:] ** 2)
    rerror_means.append(np.abs(model_mean - original_mean))
    rerror_vars.append(np.abs((model_var - original_var) / original_var))
    p += 1

plt.figure()
plt.plot(ps, rerror_means, "o-", label=r"$\vert \hat\mu - \mu \vert$")
plt.plot(
    ps, rerror_vars, "o-", label=r"$\vert \frac{\hat\sigma^2}{\sigma^2} - 1 \vert$"
)
plt.yscale("log")
plt.xlabel("Truncation degree p")
plt.ylabel("Relative error")
plt.title(f"Relative Mean, Variance Error vs. Increasing p (fixed n={N})")
plt.legend()
plt.grid(True)

# 2.6.3: Fixed p, increasing n
p = 2
alphas = create_alphas(NUM_VARIABLES, p)
P = alphas.shape[0]
Ns = range(P, 4 * P, 5)
rerror_means = []
rerror_vars = []
for N in Ns:
    Y, nominalX, idx_attr_map = get_dataset(N, ModelType.CALL)
    Psi = compute_psi(nominalX, p, idx_attr_map)
    coeff, _, _, _ = lstsq(Psi.T, Y, lapack_driver="gelsy")  # type: ignore
    Y_surrogate = Psi.T @ coeff
    model_mean = coeff[0]
    model_var = np.sum(coeff[1:] ** 2)
    rerror_means.append(np.abs(model_mean - original_mean))
    rerror_vars.append(np.abs((model_var - original_var) / original_var))

plt.figure()
plt.plot(Ns, rerror_means, "o-", label=r"$\vert \hat\mu - \mu \vert$")
plt.plot(
    Ns, rerror_vars, "o-", label=r"$\vert \frac{\hat\sigma^2}{\sigma^2} - 1 \vert$"
)
plt.yscale("log")
plt.xlabel("Dataset size N")
plt.ylabel("Relative error")
plt.title(f"Relative Mean, Variance Error vs. Increasing N (fixed p={p})")
plt.legend()
plt.grid(True)
plt.show()
