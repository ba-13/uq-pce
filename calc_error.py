import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from uq_pce.pce.pce import compute_psi
from uq_pce.pce.create_alphas import create_alphas
from scipy.linalg import lstsq
from math import comb

from uq_pce.model.model import get_dataset, INPUT_DISTRIBUTION, ModelType, NUM_VARIABLES


class PCETester:
    def __init__(self, model_func, compute_psi_func, get_dataset_func, NUM_VARIABLES):
        self.model_func = model_func
        self.compute_psi = compute_psi_func
        self.get_dataset = get_dataset_func
        self.NUM_VARIABLES = NUM_VARIABLES

    def compute_errors(self, X_train, Y_train, X_test, Y_test, P, idx_attr_map):
        # Compute Psi matrix for training data
        Psi = self.compute_psi(X_train, P, idx_attr_map)
        # Solve least-squares problem to obtain coefficients
        coeff, _, _, _ = lstsq(Psi.T, Y_train, lapack_driver="gelsy")  # type: ignore
        Y_surrogate = Psi.T @ coeff

        # Compute relative empirical error
        e_emp = np.mean((Y_train - Y_surrogate) ** 2) / np.var(Y_train)

        # Compute surrogate prediction for test data
        Psi_test = self.compute_psi(X_test, P, idx_attr_map)
        Y_surrogate_test = Psi_test.T @ coeff
        # Compute relative validation error
        e_val = np.mean((Y_test - Y_surrogate_test) ** 2) / np.var(Y_test)

        # Compute leave-one-out (LOO) error using shortcut formula
        H = Psi.T @ np.linalg.inv(Psi @ Psi.T) @ Psi
        diag_H = np.diag(H)
        residual = (Y_train - Y_surrogate) / (1 - diag_H)
        e_loo_s = np.mean(residual**2)
        var_loo = np.var(Y_train, ddof=1) * len(Y_train) / (len(Y_train) - 1)
        e_loo_s /= var_loo

        # Compute leave-one-out (LOO) error without shortcut formula
        e_loo_l = 0.0
        for i in range(len(Y_train)):
            X_train_loo = np.delete(X_train, i, axis=0)
            Y_train_loo = np.delete(Y_train, i, axis=0)
            Psi_loo = self.compute_psi(X_train_loo, P, idx_attr_map)
            coeff_loo, _, _, _ = lstsq(Psi_loo.T, Y_train_loo, lapack_driver="gelsy")  # type: ignore

            psi_i = compute_psi(X_train[i : i + 1, :], P, idx_attr_map)
            Y_pred_i = psi_i.T @ coeff_loo
            e_loo_l += ((Y_train[i] - Y_pred_i) ** 2).item()

        e_loo_l /= len(Y_train)
        e_loo_l /= var_loo

        return e_emp, e_val, e_loo_s, e_loo_l

    def test_fixed_n_increasing_p(self, N_train, N_val, max_p):
        # Fix number of training points, increase total degree p
        Y_train, X_train, idx_attr_map = self.get_dataset(N_train, self.model_func)
        Y_test, X_test, _ = self.get_dataset(N_val, self.model_func)

        p_values, e_emp_list, e_val_list, e_loo_s_list, e_loo_l_list = (
            [],
            [],
            [],
            [],
            [],
        )

        for p in range(1, max_p + 1):
            # Compute number of basis functions P
            P = create_alphas(self.NUM_VARIABLES, p).shape[0]
            if P > N_train:
                break
            e_emp, e_val, e_loo_s, e_loo_l = self.compute_errors(
                X_train, Y_train, X_test, Y_test, p, idx_attr_map
            )
            p_values.append(p)
            e_emp_list.append(e_emp)
            e_val_list.append(e_val)
            e_loo_s_list.append(e_loo_s)
            e_loo_l_list.append(e_loo_l)

        # Plot errors vs polynomial degree p
        plt.figure()
        plt.plot(p_values, e_emp_list, "o-", label="Empirical error")
        plt.plot(p_values, e_val_list, "o-", label="Validation error")
        plt.plot(p_values, e_loo_s_list, "o-", label="LOO error with shortcut")
        plt.plot(p_values, e_loo_l_list, "o-", label="LOO error")
        plt.yscale("log")
        plt.xlabel("Polynomial degree p")
        plt.ylabel("Relative error")
        plt.title(f"Errors vs. Increasing p (fixed n={N_train})")
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.legend()
        plt.grid(True)
        plt.show()

    def test_fixed_p_increasing_n(self, p, n_start, n_end, n_steps, N_val):
        # Fix polynomial degree p, increase number of training samples n
        n_values = np.linspace(n_start, n_end, n_steps, dtype=int)
        e_emp_list, e_val_list, e_loo_s_list, e_loo_l_list = [], [], [], []

        for n in n_values:
            Y_train, X_train, idx_attr_map = self.get_dataset(n, self.model_func)
            Y_test, X_test, _ = self.get_dataset(N_val, self.model_func)

            e_emp, e_val, e_loo_s, e_loo_l = self.compute_errors(
                X_train, Y_train, X_test, Y_test, p, idx_attr_map
            )
            e_emp_list.append(e_emp)
            e_val_list.append(e_val)
            e_loo_s_list.append(e_loo_s)
            e_loo_l_list.append(e_loo_l)

        # Plot errors vs training size n
        plt.figure()
        plt.plot(n_values[: len(e_emp_list)], e_emp_list, label="Empirical error")
        plt.plot(n_values[: len(e_val_list)], e_val_list, label="Validation error")
        plt.plot(
            n_values[: len(e_loo_s_list)], e_loo_s_list, label="LOO error with shortcut"
        )
        plt.plot(n_values[: len(e_loo_l_list)], e_loo_l_list, label="LOO error")
        plt.yscale("log")
        plt.xlabel("Training size n")
        plt.ylabel("Relative error")
        plt.title(f"Errors vs. Increasing n (fixed p={p}, P ={n_start})")
        plt.legend()
        plt.grid(True)

        # Add inset plot
        ax_inset = inset_axes(plt.gca(), width="40%", height="40%", loc="right")

        n_inset_min = int(np.ceil(1.5 * n_start))

        # Filter data for inset
        n_inset_values = [n for n in n_values if n >= n_inset_min]
        i_start = list(n_values).index(n_inset_values[0])
        e_emp_inset = e_emp_list[i_start:]
        e_val_inset = e_val_list[i_start:]
        e_loo_s_inset = e_loo_s_list[i_start:]
        e_loo_l_inset = e_loo_l_list[i_start:]

        # Plot inset
        plt.plot(n_inset_values, e_emp_inset, label="Empirical error")
        plt.plot(n_inset_values, e_val_inset, label="Validation error")
        plt.plot(n_inset_values, e_loo_s_inset, label="LOO error with shortcut")
        plt.plot(n_inset_values, e_loo_l_inset, label="LOO error")
        ax_inset.set_yscale("log")
        ax_inset.set_title("Inset: n > 1.5×P", fontsize=8)
        ax_inset.grid(True)
        plt.show()


if __name__ == "__main__":
    tester = PCETester(
        model_func=ModelType.CALL,
        compute_psi_func=compute_psi,
        get_dataset_func=get_dataset,
        NUM_VARIABLES=NUM_VARIABLES,
    )

    N_train = 40  # number of experimental samples
    N_test = 1000  # number of testing samples
    p_init = 2  # original total degree
    M_init = 4  # number of input parameters
    P_init = comb(M_init + p_init, p_init)  # calculate P

    # Run test with fixed n and increasing p
    tester.test_fixed_n_increasing_p(N_train=N_train, N_val=N_test, max_p=10)

    # Run test with fixed p and increasing n from P to 4P
    tester.test_fixed_p_increasing_n(
        p=p_init, n_start=P_init, n_end=4 * P_init, n_steps=10, N_val=N_test
    )
