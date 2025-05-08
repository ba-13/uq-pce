import numpy as np
from src.model.black_scholes import Option
import math as m
from scipy.stats import qmc

def create_alphas(n_inputs,n):
     return np.array(list(filter(lambda x: sum(x) <= n, np.ndindex((n + 1,) * n_inputs))))

def eval_legendre(X, deg):
    """
    Evaluate the normalized Legendre polynomial of degree 'deg'
    for values in array X.
    """
    if deg < 0:
        raise ValueError("The degree should be at least 0!")

    # Compute the unnormalized Legendre polynomials
    if deg == 0:
        P = np.ones_like(X)
    elif deg == 1:
        P = X
    else:
        P_nminus1 = np.ones_like(X)
        P_n = X

        for n in range(1, deg):
            coef1 = (2 * n + 1) / (n + 1)
            coef2 = n / (n + 1)
            P_nplus1 = coef1 * X * P_n - coef2 * P_nminus1
            P_nminus1 = P_n
            P_n = P_nplus1

        P = P_nplus1

    # Normalize the Legendre polynomial
    P = m.sqrt(2 * deg + 1) * P

    return P

def myCustomModel(X: np.ndarray):

        n = X.shape[0]
        K = 100.0
        T = 1.0 #expire time is fixed usually in reality
        Y = np.zeros(n)

        for i in range(n):
                s, sigma, r, tau = X[i]
                t =  T - tau

                option = Option()
                option.sigma = sigma
                option.r = r
                option.K = K
                option.T = T

                Y[i] = option.call_option_price(s,t)

        return Y


seed = 42
n_val = 1000000
dim = 4
p =4

boundary={'S': [50, 150],'sigma': [0.1, 0.5],'r': [0.01, 0.05], 'tau': [0.01, 1.0]}




print("Generating and saving validation set...")
np.random.seed(seed)  
X_val = np.random.uniform(0, 1, size=(n_val, dim))  
X_real_val = np.zeros_like(X_val)
for i, (param, (min, max)) in enumerate(boundary.items()):
	X_real_val[:, i] = min + (max - min) * X_val[:, i]
np.save("X_real_val", X_real_val)


print("evaluating validation with analytical formula")
y_val = myCustomModel(X_real_val)
np.save("y_val", y_val)



print("evaluating validation with surrogate model")
### from here below is the same as pce.py
X_legendre_val = np.zeros_like(X_real_val)
for i, (param, (min, max)) in enumerate(boundary.items()):
    X_legendre_val[:, i] = 2 * (X_real_val[:, i] - min) / (max - min) - 1

alphas = create_alphas(dim, p)
P = alphas.shape[0]  

n_samples = 1000

sampler = qmc.LatinHypercube(d=dim)
X_lhm =sampler.random(n=n_samples) 

# transform to uniform distributions of real ranges
X_real= np.zeros_like(X_lhm)
for i, (param, (min, max)) in enumerate(boundary.items()):
    X_real[:, i] = min + (max - min) * X_lhm[:, i]

X_legendre = np.zeros_like(X_real)
for i, (param, (min,max)) in enumerate(boundary.items()):
    X_legendre[:, i] = 2 * (X_real[:, i] - min) / (max-min) - 1

Y = myCustomModel(X_real)
uni_poly = []

for i in range(dim):
    polys_i = []
    for k in range(p + 1):
        psi_k = eval_legendre(X_legendre[:, i], k)
        polys_i.append(psi_k)
    uni_poly.append(polys_i)



Psi = np.ones((n_samples, P))
 
for j in range(P):
    alpha = alphas[j]
    for i in range(dim):
        Psi[:, j] *= uni_poly[i][alpha[i]]



PsiT_Psi = Psi.T @ Psi  
PsiT_Psi_inv = np.linalg.inv(PsiT_Psi)
PsiT_y = Psi.T @ Y
c = PsiT_Psi_inv @ PsiT_y
### end of the section from pce.py

### validation using the validation set
uni_poly_val = []
for i in range(dim):
    polys_i = []
    for k in range(p + 1):
        psi_k = eval_legendre(X_legendre_val[:, i], k)
        polys_i.append(psi_k)
    uni_poly_val.append(polys_i)

Psi_val = np.ones((n_val, P))

for j in range(P):
    alpha = alphas[j]
    for i in range(dim):
        Psi_val[:, j] *= uni_poly_val[i][alpha[i]]


y_pred_val = Psi_val @ c
np.save("y_surrogate_val", y_pred_val)
