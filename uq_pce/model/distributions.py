from math import factorial, sqrt
import numpy as np
import numpy.typing as npt
from scipy.stats import Normal, Uniform


class Distribution:
    def sample(self, N: int) -> npt.NDArray[np.float64]:
        raise NotImplementedError("This is an abstract class method")

    def isoprobabilistic(self, numbers: npt.NDArray[np.float64]):
        raise NotImplementedError("This is an abstract class method")

    def isoprobabilistic_nominal(self, numbers: npt.NDArray[np.float64]):
        raise NotImplementedError("This is an abstract class method")

    @staticmethod
    def evaluate_polynomial(
        X: npt.NDArray[np.float64], deg: int
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError("This is an abstract class method")


class FixedDistribution(Distribution):
    def __init__(self, value: float):
        self.value = value

    def sample(self, N: int) -> npt.NDArray[np.float64]:
        return np.full(N, self.value)


class UniformDistribution(Distribution):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self.sampler = Uniform(a=low, b=high)

    def sample(self, N: int) -> npt.NDArray[np.float64]:
        return self.sampler.sample(N)

    def isoprobabilistic(self, numbers: npt.NDArray[np.float64]):
        return self.sampler.icdf(numbers)

    def isoprobabilistic_nominal(self, numbers: npt.NDArray[np.float64]):
        return Uniform(a=-1, b=1).icdf(numbers)

    @staticmethod
    def evaluate_polynomial(
        X: npt.NDArray[np.float64], deg: int
    ) -> npt.NDArray[np.float64]:
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

            P = P_n

        # Normalize the Legendre polynomial
        P = sqrt(2 * deg + 1) * P

        return P


class NormalDistribution(Distribution):
    def __init__(self, mean: float, stddev: float):
        self.mean = mean
        self.stddev = stddev
        self.sampler = Normal(mu=mean, sigma=stddev)

    def sample(self, N: int) -> npt.NDArray[np.float64]:
        return self.sampler.sample(N)

    def isoprobabilistic(self, numbers: npt.NDArray[np.float64]):
        return self.sampler.icdf(numbers)

    def isoprobabilistic_nominal(self, numbers: npt.NDArray[np.float64]):
        return Normal(mu=0, sigma=1).icdf(numbers)

    @staticmethod
    def evaluate_polynomial(
        X: npt.NDArray[np.float64], deg: int
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate the normalized Hermite polynomial of degree 'deg'
        for values in array X.
        """
        if deg < 0:
            raise ValueError("The degree should be at least 0!")

        # Compute the unnormalized Hermite polynomials
        if deg == 0:
            P = np.ones_like(X)
        elif deg == 1:
            P = X
        else:
            P_nminus1 = np.ones_like(X)
            P_n = X

            for n in range(1, deg):
                P_nplus1 = X * P_n - n * P_nminus1
                P_nminus1 = P_n
                P_n = P_nplus1

            P = P_n

        # Normalize the Hermite polynomial
        P = P / sqrt(factorial(deg))

        return P
