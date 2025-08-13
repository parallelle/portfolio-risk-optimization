# Portfolio Risk & Optimization (All simulated Data)

This is a small, self-contained project that **simulates 5 equities**, builds **20,000 random portfolios**, plots the **efficient frontier**, and identifies the **Max Sharpe** portfolio. It also computes **1-day 95% VaR** and **CVaR** for the Max Sharpe portfolio.

## Why this exists
- Demonstrates the ability to turn **Modern Portfolio Theory** into working code.
- Shows comfort with **Python, pandas, numpy, matplotlib** and basic **risk modeling**.
- Makes a clean artifact set for resumes: CSV (top portfolios), PNG (frontier), JSON (summary).

## How to run
```bash
# (optional) create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install deps
pip install -r requirements.txt

# run
python portfolio_optimization.py
```

Outputs will appear in the repo root:
- `efficient_frontier.png` — scatter of portfolios, Max Sharpe highlighted
- `max_sharpe_top10.csv` — top-10 portfolios by Sharpe
- `max_sharpe_summary.json` — key metrics and weights

## Key methods
- Simulate daily returns for 5 assets using multivariate normal with realistic **means, volatilities, correlations**.
- Compute annualized **return**, **volatility**, and **Sharpe** for each random portfolio.
- Estimate **VaR** and **CVaR** for the Max Sharpe portfolio.
- Save artifacts for quick review.

## Tech
- Python 3.10+
- pandas, numpy, matplotlib

## Notes
- Results are deterministic given the fixed random seed; you can change the `seed` argument in `main()` to explore different outcomes.
- No external data sources required.
