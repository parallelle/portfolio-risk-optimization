# portfolio_optimization.py
# Simulated Quantitative Risk Model for a 5-Asset Portfolio
# - Generates realistic daily return series for 5 synthetic equities over ~5 years
# - Computes annualized return/volatility, runs 20,000 random portfolios
# - Identifies the Max Sharpe portfolio
# - Calculates 1-day 95% VaR and CVaR for that portfolio
# - Saves outputs: top-10 CSV, summary JSON, efficient frontier PNG

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(seed: int = 11, n_years: int = 5, n_assets: int = 5, n_ports: int = 20000):
    np.random.seed(seed)

    # ---- 1) Simulate daily returns for synthetic assets ----
    tickers = [f"EQ{i+1}" for i in range(n_assets)]
    T = 252 * n_years

    # Annual parameters (plausible for equities)
    annual_means = np.array([0.12, 0.10, 0.14, 0.08, 0.11])[:n_assets]
    annual_vols  = np.array([0.25, 0.20, 0.30, 0.18, 0.22])[:n_assets]

    # Correlation matrix (moderate positive correlations)
    corr = np.array([
        [1.00, 0.65, 0.55, 0.45, 0.50],
        [0.65, 1.00, 0.50, 0.40, 0.45],
        [0.55, 0.50, 1.00, 0.35, 0.40],
        [0.45, 0.40, 0.35, 1.00, 0.30],
        [0.50, 0.45, 0.40, 0.30, 1.00],
    ])[:n_assets,:n_assets]

    mu_d = annual_means / 252.0
    sigma_d = annual_vols / np.sqrt(252.0)
    cov_d = np.outer(sigma_d, sigma_d) * corr  # daily covariance

    returns = np.random.multivariate_normal(mean=mu_d, cov=cov_d, size=T)
    returns = pd.DataFrame(returns, columns=tickers)

    # ---- 2) Annualized stats ----
    mean_returns_ann = returns.mean() * 252
    cov_ann = returns.cov() * 252

    # ---- 3) Random portfolio simulation (efficient frontier) ----
    results = np.zeros((3 + len(tickers), n_ports))

    for i in range(n_ports):
        w = np.random.random(len(tickers))
        w /= w.sum()
        port_ret = float(np.dot(w, mean_returns_ann))
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_ann, w))))
        sharpe = port_ret / port_vol if port_vol != 0 else np.nan
        results[0, i] = port_ret
        results[1, i] = port_vol
        results[2, i] = sharpe
        results[3:, i] = w

    columns = ["Return", "Volatility", "Sharpe"] + tickers
    frontier = pd.DataFrame(results.T, columns=columns)

    # Max Sharpe portfolio
    max_idx = frontier["Sharpe"].idxmax()
    max_port = frontier.loc[max_idx]

    # ---- 4) Risk metrics for Max Sharpe portfolio ----
    weights = max_port[tickers].values
    port_daily = (returns @ weights)

    VaR_95 = np.percentile(port_daily, 5)          # 5th percentile (1-day, 95% VaR)
    CVaR_95 = port_daily[port_daily <= VaR_95].mean()  # average of the worst 5%

    # ---- 5) Save artifacts ----
    top10 = frontier.sort_values("Sharpe", ascending=False).head(10)
    top10.to_csv("max_sharpe_top10.csv", index=False)

    summary = {
        "Annualized Return": float(max_port["Return"]),
        "Annualized Volatility": float(max_port["Volatility"]),
        "Sharpe": float(max_port["Sharpe"]),
        "Weights": {t: float(max_port[t]) for t in tickers},
        "VaR_95_1day": float(VaR_95),
        "CVaR_95_1day": float(CVaR_95),
        "Seed": seed,
        "Years": n_years,
        "PortsSimulated": n_ports
    }
    with open("max_sharpe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(9,6))
    plt.scatter(frontier["Volatility"], frontier["Return"], alpha=0.4)
    plt.scatter(max_port["Volatility"], max_port["Return"], s=60)
    plt.xlabel("Volatility (Annualized Std Dev)")
    plt.ylabel("Return (Annualized Mean)")
    plt.title("Simulated Efficient Frontier (Max Sharpe Highlighted)")
    plt.grid(True)
    plt.savefig("efficient_frontier.png", bbox_inches="tight")
    # plt.show()  # uncomment to display when running locally

    # ---- 6) Print summary ----
    print("== Max Sharpe Portfolio ==")
    print(f"Return   : {summary['Annualized Return']:.4f}")
    print(f"Volatility: {summary['Annualized Volatility']:.4f}")
    print(f"Sharpe   : {summary['Sharpe']:.4f}")
    print("Weights  :")
    for t, w in summary["Weights"].items():
        print(f"  {t}: {w:.4f}")
    print(f"VaR 95%  : {summary['VaR_95_1day']:.4%}")
    print(f"CVaR 95% : {summary['CVaR_95_1day']:.4%}")
    print("\nArtifacts saved: max_sharpe_top10.csv, max_sharpe_summary.json, efficient_frontier.png")

if __name__ == "__main__":
    main()
