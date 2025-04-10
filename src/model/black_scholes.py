import numpy as np
import scipy.stats as stats


class Option:
    """Option pricing model
    Assuming current time is 0, option expires at T;
    and we need to price the option (either call or put) between 0 and T.
    To avoid arbitrage, market tends towards keeping call option price + strike price = asset price
    Assumptions:
    - option price is zero for a zero valued asset
    - if asset price is high enough, call option price would just be price of asset minus the cost to buy the stock
    - just before maturity, call option price would be 
    max{profit of buying asset, 0}
    More assumptions: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Fundamental_hypotheses
    """
    r: np.float32  # riskfree interest rate
    sigma: np.float32  # stddev of stock's returns
    T: np.float32  # time of expiry of this option
    K: np.float32  # strike price

    def call_option_price(self, s: np.float32, t: np.float32):
        """Generates call price at given underlying asset price
        at some time t

        Args:
            s (np.float32): asset price
            t (np.float32): prediction time
        """
        tau = self.T - t
        den = self.sigma * np.sqrt(tau)
        dplus = (np.log(s / self.K) + (self.r + self.sigma**2 / 2) * tau) / den
        dminus = dplus - den
        return (stats.norm.cdf(dplus) * s) - (
            stats.norm.cdf(dminus) * self.K * np.exp(-self.r * tau)
        )

    def put_option_price(self, s: np.float32, t: np.float32):
        """Generates put option price at given underlying asset price at some time t

        Args:
            s (np.float32): asset price
            t (np.float32): prediction time
        """
        tau = self.T - t
        den = self.sigma * np.sqrt(tau)
        dplus = (np.log(s / self.K) + (self.r + self.sigma**2 / 2) * tau) / den
        dminus = dplus - den
        return (stats.norm.cdf(-dminus) * self.K * np.exp(-self.r * tau)) - (
            stats.norm.cdf(-dplus) * s
        )
