from src.model.black_scholes import Option
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import pandas as pd

from math import sqrt




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
    P = sqrt(2 * deg + 1) * P

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


n_samples = 10000
dim = 4
p = 2


boundary={'S': [50, 150],'sigma': [0.1, 0.5],'r': [0.01, 0.05], 'tau': [0.01, 1.0]}

# select points using Latin hypercube method, in U~[0,1]^4
sampler = qmc.LatinHypercube(d=dim)
X_lhm =sampler.random(n=n_samples) 

# transform to uniform distributions of real ranges
X_real= np.zeros_like(X_lhm)
for i, (param, (min, max)) in enumerate(boundary.items()):
    X_real[:, i] = min + (max - min) * X_lhm[:, i]


Y = myCustomModel(X_real) 

plt.figure( figsize = (10, 8) )
for i, param in enumerate(boundary.keys()):
    plt.subplot(2, 2, i+1)
    plt.hist(X_real[:, i], bins=20, density=True)

#plt.show()


# isoprobabilistic transformation to unit hypercube [0, 1] for least square
X_lhm_back = np.zeros_like(X_real)
for i, (param, (min, max)) in enumerate(boundary.items()):

    X_lhm_back[:, i] = (X_real[:, i] - min) / (max -min)
print(X_lhm_back)

plt.figure(figsize=(10, 8))
for i, param in enumerate(boundary.keys()):
    plt.subplot(2, 2, i+1)
    plt.hist(X_lhm_back[:, i], bins=20, density=True)

#plt.show()


# for Legendre 
X_legendre = np.zeros_like(X_real)

for i, (param, (min, max)) in enumerate(boundary.items()):

    X_legendre[:, i] = 2 * (X_real[:, i] - min) / (max - min) - 1


uni_poly = [] 

# generating univariate polynomial for all 4 dimensions
for i in range(dim):
    polys_i = []
    for k in range(p + 1):
        psi_k = eval_legendre(X_legendre[:, i], k)
        polys_i.append(psi_k)
    uni_poly.append(polys_i)


#print( uni_poly)
alphas = create_alphas(dim, p)
P = alphas.shape[0]  


#print(alphas)

Psi = np.ones((n_samples, P))  

# truncation 
for j in range(P):
    alpha = alphas[j]
    for i in range(dim):
        Psi[:, j] *= uni_poly[i][alpha[i]]


# computing coefficients
PsiT_Psi = Psi.T @ Psi  
PsiT_Psi_inv = np.linalg.inv(PsiT_Psi)
PsiT_y = Psi.T @ Y
c = PsiT_Psi_inv @ PsiT_y



# y-y plot

Y_surrogate = Psi @ c 
plt.figure(figsize=(8, 6))
plt.scatter(Y, Y_surrogate, alpha=0.5)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')
plt.xlabel('True Option Prices (Y)')
plt.ylabel('Surrogate Prices')
plt.title('PCE vs Real')
plt.grid(True)
plt.show()
