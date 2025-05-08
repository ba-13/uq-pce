from enum import Enum
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


def call_option_price(
    s: npt.NDArray[np.float32],
    t: npt.NDArray[np.float32],
    T: npt.NDArray[np.float32],
    K: npt.NDArray[np.float32],
    sigma: npt.NDArray[np.float32],
    r: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Generates call price at given underlying asset price
    at some time t

    Args:
        s (npt.NDArray[np.float32]): asset price
        t (npt.NDArray[np.float32]): prediction time
        T (npt.NDArray[np.float32]): maturity time
        K (npt.NDArray[np.float32]): strike price
        sigma (npt.NDArray[np.float32]): volatility, stddev of asset price
        r (npt.NDArray[np.float32]): risk-free interest rate
    """
    tau = T - t
    den = sigma * np.sqrt(tau)
    dplus = (np.log(s / K) + (r + sigma**2 / 2) * tau) / den
    dminus = dplus - den
    return (stats.norm.cdf(dplus) * s) - (stats.norm.cdf(dminus) * K * np.exp(-r * tau))


def put_option_price(
    s: npt.NDArray[np.float32],
    t: npt.NDArray[np.float32],
    T: npt.NDArray[np.float32],
    K: npt.NDArray[np.float32],
    sigma: npt.NDArray[np.float32],
    r: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Generates put option price at given underlying asset price at some time t

    Args:
        s (npt.NDArray[np.float32]): asset price
        t (npt.NDArray[np.float32]): prediction time
        T (npt.NDArray[np.float32]): maturity time
        K (npt.NDArray[np.float32]): strike price
        sigma (npt.NDArray[np.float32]): volatility
        r (npt.NDArray[np.float32]): risk-free interest rate
    """
    tau = T - t
    den = sigma * np.sqrt(tau)
    dplus = (np.log(s / K) + (r + sigma**2 / 2) * tau) / den
    dminus = dplus - den
    return (stats.norm.cdf(-dminus) * K * np.exp(-r * tau)) - (
        stats.norm.cdf(-dplus) * s
    )
