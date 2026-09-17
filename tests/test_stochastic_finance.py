import numpy as np

from src.stochastic_finance import black_scholes_call, brownian_path, monte_carlo_call


def test_brownian_shapes_and_seed():
    t1, w1, dw1 = brownian_path(n_steps=10, seed=7)
    t2, w2, dw2 = brownian_path(n_steps=10, seed=7)
    assert (len(t1), len(w1), len(dw1)) == (11, 11, 10)
    assert w1[0] == 0
    assert np.array_equal(w1, w2) and np.array_equal(dw1, dw2)


def test_monte_carlo_agrees_with_closed_form_within_sampling_error():
    exact = black_scholes_call(100, 100, 1, 0.05, 0.2)
    estimate, standard_error = monte_carlo_call(100, 100, 1, 0.05, 0.2, 200_000, seed=42)
    assert abs(estimate - exact) < 4 * standard_error

