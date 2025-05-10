import numpy as np
from scipy import stats
import numpy.typing as npt

"""Option pricing model
Assuming current time is 0, option expires at T;
and we need to price the option (either call or put) between 0 and T.
To avoid arbitrage, market tends towards keeping call option price + strike price = asset price
Assumptions:
- r is yearly risk-free interest rate
- sigma is yearly volatility
- T is time in years (to keep dimensions consistent)
- option price is zero for a zero valued asset
- if asset price is high enough, call option price would just be price of asset minus the cost to buy the stock
- just before maturity, call option price would be max{profit of buying asset, 0}
More assumptions: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Fundamental_hypotheses
"""


def call_option_price(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Args
    --------
        s (npt.NDArray[np.float64]): asset price
        t (npt.NDArray[np.float64]): prediction time
        T (npt.NDArray[np.float64]): maturity time
        K (npt.NDArray[np.float64]): strike price
        sigma (npt.NDArray[np.float64]): volatility, stddev of asset price
        r (npt.NDArray[np.float64]): risk-free interest rate
    Returns
    --------
        npt.NDArray[np.float64]: call option price
    """
    s     = X[..., 0]
    t     = X[..., 1]
    T     = X[..., 2]
    K     = X[..., 3]
    sigma = X[..., 4]
    r     = X[..., 5]

    tau = np.maximum(T - t, 0.0)
    sqrt_tau = np.sqrt(tau, where=(tau>0), out=np.zeros_like(tau))

    d1 = (np.log(s / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    return s * stats.norm.cdf(d1) - K * np.exp(-r * tau) * stats.norm.cdf(d2)


def put_option_price(X: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Args
    --------
        s (npt.NDArray[np.float64]): asset price
        t (npt.NDArray[np.float64]): prediction time
        T (npt.NDArray[np.float64]): maturity time
        K (npt.NDArray[np.float64]): strike price
        sigma (npt.NDArray[np.float64]): volatility
        r (npt.NDArray[np.float64]): risk-free interest rate
    Returns
    --------
        npt.NDArray[np.float64]: put option price
    """
    s     = X[..., 0]
    t     = X[..., 1]
    T     = X[..., 2]
    K     = X[..., 3]
    sigma = X[..., 4]
    r     = X[..., 5]
    
    tau = np.maximum(T - t, 0.0)
    sqrt_tau = np.sqrt(tau, where=(tau>0), out=np.zeros_like(tau))

    d1 = (np.log(s / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    return K * np.exp(-r * tau) * stats.norm.cdf(-d2) - s * stats.norm.cdf(-d1)
