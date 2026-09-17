"""Small, reproducible building blocks used by the stochastic-finance notebook."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def brownian_path(horizon: float = 1.0, n_steps: int = 252, seed: int | None = None):
    """Return time grid, Brownian path (including W_0), and increments."""
    if horizon <= 0 or n_steps < 1:
        raise ValueError("horizon and n_steps must be positive")
    rng = np.random.default_rng(seed)
    dt = horizon / n_steps
    increments = rng.normal(0.0, math.sqrt(dt), n_steps)
    path = np.concatenate(([0.0], np.cumsum(increments)))
    return np.linspace(0.0, horizon, n_steps + 1), path, increments


def black_scholes_call(spot: float, strike: float, maturity: float, rate: float, volatility: float) -> float:
    """Price a European call under the Black–Scholes model."""
    if min(spot, strike, maturity, volatility) <= 0:
        raise ValueError("spot, strike, maturity, and volatility must be positive")
    root_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    normal = NormalDist()
    return spot * normal.cdf(d1) - strike * math.exp(-rate * maturity) * normal.cdf(d2)


def monte_carlo_call(spot: float, strike: float, maturity: float, rate: float, volatility: float,
                     n_paths: int = 100_000, seed: int | None = 42) -> tuple[float, float]:
    """Estimate a European call price and its Monte Carlo standard error."""
    if n_paths < 2:
        raise ValueError("n_paths must be at least two")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n_paths)
    terminal = spot * np.exp((rate - 0.5 * volatility**2) * maturity + volatility * math.sqrt(maturity) * shocks)
    discounted = math.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0)
    return float(discounted.mean()), float(discounted.std(ddof=1) / math.sqrt(n_paths))

