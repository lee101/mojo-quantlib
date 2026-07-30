"""Black-76, Black-Scholes Greeks, implied volatility, and American CRR."""

from __future__ import annotations

import math
import subprocess
from collections import namedtuple

import numpy as np

from ._lib import addr, f64, i64, lib


class Option:
    Call = 1
    Put = -1


class PlainVanillaPayoff:
    def __init__(self, option_type, strike):
        self._option_type = int(option_type)
        if self._option_type not in (Option.Call, Option.Put):
            raise ValueError("option type must be Option.Call or Option.Put")
        self._strike = float(strike)
        if not math.isfinite(self._strike):
            raise ValueError("strike must be finite")

    def optionType(self):
        return self._option_type

    def strike(self):
        return self._strike

    def __call__(self, price):
        sign = 1.0 if self._option_type > 0 else -1.0
        return max(sign * (float(price) - self._strike), 0.0)


BlackScholesResult = namedtuple(
    "BlackScholesResult", "price delta gamma vega theta rho"
)


def _broadcast(option_type, *values):
    arrays = np.broadcast_arrays(
        i64(option_type),
        *(f64(x) for x in values),
    )
    shape = arrays[0].shape
    scalar = shape == ()
    arrays = [np.ascontiguousarray(x.reshape(-1)) for x in arrays]
    if np.any((arrays[0] != Option.Call) & (arrays[0] != Option.Put)):
        raise ValueError("option type must be Option.Call or Option.Put")
    return arrays, shape, scalar


def _broadcast_compact(option_type, *values):
    source = [
        i64(option_type),
        *(f64(x) for x in values),
    ]
    shape = np.broadcast_shapes(*(x.shape for x in source))
    scalar = shape == ()
    arrays = []
    scalar_mask = 0
    for index, array in enumerate(source):
        if array.size == 1:
            scalar_mask |= 1 << index
            arrays.append(np.ascontiguousarray(array.reshape(-1)))
        else:
            arrays.append(np.ascontiguousarray(np.broadcast_to(array, shape).reshape(-1)))
    if np.any((arrays[0] != Option.Call) & (arrays[0] != Option.Put)):
        raise ValueError("option type must be Option.Call or Option.Put")
    size = math.prod(shape) if shape else 1
    return arrays, shape, scalar, scalar_mask, size


def _finish(result, shape, scalar):
    shaped = result.reshape(shape)
    return float(shaped) if scalar else shaped


def _gpu_memory_available():
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        values = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return proc.returncode == 0 and bool(values) and int(values[0]) >= 4000
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def blackFormula(
    optionType, strike, forward, stdDev, discount=1.0, displacement=0.0
):
    arrays, shape, scalar = _broadcast(
        optionType, strike, forward, stdDev, discount, displacement
    )
    types, strike, forward, stddev, discount, displacement = arrays
    if np.any(strike + displacement <= 0) or np.any(forward + displacement <= 0):
        raise ValueError("strike+displacement and forward+displacement must be positive")
    if np.any(stddev < 0) or np.any(discount <= 0):
        raise ValueError("standard deviation must be non-negative and discount positive")
    result = np.empty(types.size, dtype=np.float64)
    lib().mql_black_formula(
        *(addr(x) for x in arrays), addr(result), result.size
    )
    return _finish(result, shape, scalar)


def black_formula(*args, **kwargs):
    return blackFormula(*args, **kwargs)


def blackFormulaImpliedStdDev(
    optionType, strike, forward, blackPrice, discount=1.0, displacement=0.0,
    guess=None, accuracy=1e-6, maxIterations=100, device="cpu",
):
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    if not math.isfinite(float(accuracy)) or accuracy <= 0:
        raise ValueError("accuracy must be positive and finite")
    if isinstance(maxIterations, (bool, np.bool_)) or not isinstance(
        maxIterations, (int, np.integer)
    ) or maxIterations < 1:
        raise ValueError("maxIterations must be a positive integer")
    arrays, shape, scalar, scalar_mask, size = _broadcast_compact(
        optionType, strike, forward, blackPrice, discount, displacement
    )
    types, strikes, forwards, prices, discounts, displacements = arrays
    if np.any(strikes + displacements <= 0) or np.any(
        forwards + displacements <= 0
    ):
        raise ValueError("strike+displacement and forward+displacement must be positive")
    if np.any(discounts <= 0) or np.any(prices < 0):
        raise ValueError("discount must be positive and price non-negative")
    result = np.empty(size, dtype=np.float64)
    failed = -1
    if (
        device == "gpu"
        and 0 < result.size
        and result.size * 56 < 2_000_000_000
        and _gpu_memory_available()
    ):
        failed = lib().mql_black_implied_stddev_gpu(
            *(addr(x) for x in arrays), addr(result), result.size, scalar_mask,
            accuracy, maxIterations,
        )
    if failed == -2:
        raise RuntimeError("GPU implied-standard-deviation execution failed")
    if failed == -1:
        failed = lib().mql_black_implied_stddev(
            *(addr(x) for x in arrays), addr(result), result.size, scalar_mask,
            accuracy, maxIterations,
        )
    if failed:
        raise RuntimeError(f"price at flat index {failed - 1} is outside Black bounds")
    return _finish(result, shape, scalar)


def black_formula_implied_std_dev(*args, **kwargs):
    return blackFormulaImpliedStdDev(*args, **kwargs)


def black_scholes_greeks(
    option_type, spot, strike, maturity, risk_free_rate, volatility,
    dividend_yield=0.0,
):
    arrays, shape, scalar = _broadcast(
        option_type, spot, strike, maturity, risk_free_rate, volatility,
        dividend_yield,
    )
    if np.any(arrays[1] <= 0) or np.any(arrays[2] <= 0):
        raise ValueError("spot and strike must be positive")
    if np.any(arrays[3] < 0) or np.any(arrays[5] < 0):
        raise ValueError("maturity and volatility must be non-negative")
    outputs = [np.empty(arrays[0].size, dtype=np.float64) for _ in range(6)]
    lib().mql_black_scholes(
        *(addr(x) for x in arrays), *(addr(x) for x in outputs), arrays[0].size
    )
    return BlackScholesResult(
        *(_finish(x, shape, scalar) for x in outputs)
    )


def black_scholes(
    option_type, spot, strike, maturity, risk_free_rate, volatility,
    dividend_yield=0.0,
):
    return black_scholes_greeks(
        option_type, spot, strike, maturity, risk_free_rate, volatility,
        dividend_yield,
    ).price


def american_option(
    option_type, spot, strike, maturity, risk_free_rate, volatility,
    dividend_yield=0.0, steps=500,
):
    if isinstance(steps, (bool, np.bool_)) or not isinstance(
        steps, (int, np.integer)
    ) or steps < 1:
        raise ValueError("steps must be a positive integer")
    arrays, shape, scalar = _broadcast(
        option_type, spot, strike, maturity, risk_free_rate, volatility,
        dividend_yield,
    )
    if np.any(arrays[1] <= 0) or np.any(arrays[2] <= 0):
        raise ValueError("spot and strike must be positive")
    if np.any(arrays[3] < 0) or np.any(arrays[5] < 0):
        raise ValueError("maturity and volatility must be non-negative")
    dt = arrays[3] / steps
    valid = (dt <= 0) | (arrays[5] <= 0)
    up = np.exp(arrays[5] * np.sqrt(np.maximum(dt, 0)))
    down = np.divide(1.0, up, out=np.ones_like(up), where=up != 0)
    probability = np.divide(
        np.exp((arrays[4] - arrays[6]) * dt) - down,
        up - down,
        out=np.full_like(up, 0.5),
        where=up != down,
    )
    if np.any(~valid & ((probability < 0) | (probability > 1))):
        raise ValueError("steps are too few for valid CRR probabilities")
    result = np.empty(arrays[0].size, dtype=np.float64)
    work = np.empty(steps + 1, dtype=np.float64)
    lib().mql_american_crr(
        *(addr(x) for x in arrays), addr(result), arrays[0].size, steps, addr(work)
    )
    return _finish(result, shape, scalar)


class BlackCalculator:
    def __init__(self, payoff, forward, stdDev, discount=1.0):
        if not hasattr(payoff, "optionType") or not hasattr(payoff, "strike"):
            raise TypeError("payoff must provide optionType() and strike()")
        self.optionType = int(payoff.optionType())
        self.strike = float(payoff.strike())
        self.forward = float(forward)
        self.stdDev = float(stdDev)
        self.discount = float(discount)
        self._sign = 1.0 if self.optionType > 0 else -1.0
        if self.stdDev > 0:
            self._d1 = math.log(self.forward / self.strike) / self.stdDev + self.stdDev / 2
            self._d2 = self._d1 - self.stdDev
        else:
            self._d1 = math.inf if self.forward > self.strike else -math.inf
            self._d2 = self._d1

    @staticmethod
    def _cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _pdf(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def value(self):
        return blackFormula(
            self.optionType, self.strike, self.forward, self.stdDev, self.discount
        )

    def deltaForward(self):
        return self._sign * self.discount * self._cdf(self._sign * self._d1)

    def delta(self, spot):
        return self.deltaForward() * self.forward / float(spot)

    def gammaForward(self):
        if self.stdDev <= 0:
            return 0.0
        return self.discount * self._pdf(self._d1) / (self.forward * self.stdDev)

    def gamma(self, spot):
        return self.gammaForward() * (self.forward / float(spot)) ** 2

    def vega(self, maturity):
        return self.discount * self.forward * self._pdf(self._d1) * math.sqrt(maturity)

    def itmCashProbability(self):
        return self._cdf(self._sign * self._d2)

    def itmAssetProbability(self):
        return self._cdf(self._sign * self._d1)
