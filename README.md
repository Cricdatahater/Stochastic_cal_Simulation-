# Stochastic Calculus & Monte Carlo Finance

Reproducible Python experiments connecting Brownian motion, Itô calculus, Monte Carlo estimation, Black–Scholes pricing, and realized volatility.

## What this repository demonstrates

- simulation of Brownian paths and stochastic integrals;
- numerical illustration of the Itô correction for \(f(W_t)=W_t^2\);
- European call pricing by both Black–Scholes and risk-neutral Monte Carlo;
- Monte Carlo confidence intervals and convergence diagnostics;
- exploratory analysis of equity returns and rolling realized volatility.

The original exploratory work is retained in [`Stochastic_cal.ipynb`](Stochastic_cal.ipynb). Reusable, testable implementations live in [`src/stochastic_finance.py`](src/stochastic_finance.py).

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
jupyter lab Stochastic_cal.ipynb
```

The deterministic unit tests do not require network access. Notebook sections that download AAPL data use `yfinance` and therefore depend on the data available at execution time.

## Core API

```python
from src.stochastic_finance import black_scholes_call, monte_carlo_call

closed_form = black_scholes_call(100, 100, 1.0, 0.05, 0.20)
estimate, standard_error = monte_carlo_call(
    100, 100, 1.0, 0.05, 0.20, n_paths=100_000, seed=42
)
```

## Methodological notes

- Monte Carlo pricing is performed under the risk-neutral geometric Brownian motion model.
- Reported uncertainty is simulation error, not model uncertainty.
- Black–Scholes assumes constant volatility, frictionless markets, and log-normal prices; the empirical volatility sections are exploratory rather than a validation of those assumptions.
- A fixed random seed makes examples reproducible, but conclusions should be checked across seeds and path counts.

## Repository structure

```text
.
├── Stochastic_cal.ipynb        # exploratory derivations and visualizations
├── src/stochastic_finance.py   # reusable numerical functions
├── tests/test_stochastic_finance.py
├── requirements.txt
└── README.md
```

## Next research steps

- compare discretization error across time-step grids;
- add variance-reduction methods and error-versus-runtime benchmarks;
- replace live-only market examples with a dated, documented data snapshot;
- add model diagnostics for non-normal returns and volatility clustering.

