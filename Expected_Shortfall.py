"""
Expected Shortfall via Monte Carlo Simulation
==============================================
Simulates a simple 5-asset macro portfolio and computes
95% Expected Shortfall (CVaR) on the daily P&L distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── Portfolio Definition ───────────────────────────────────────────────────────

PORTFOLIO = {
    #  Asset          Notional ($)   Ann. vol   Ann. drift
    "USD/JPY":       (1_000_000,    0.08,       0.02),
    "US 10Y Bond":   (2_000_000,    0.06,       0.03),
    "Gold":          (  800_000,    0.15,       0.05),
    "Crude Oil":     (  600_000,    0.30,       0.00),
    "S&P 500":       (1_500_000,    0.18,       0.10),
}

# Correlation matrix (same order as PORTFOLIO)
CORR = np.array([
    [ 1.00, -0.30,  0.20,  0.10,  0.15],
    [-0.30,  1.00,  0.35, -0.25, -0.40],
    [ 0.20,  0.35,  1.00,  0.05, -0.10],
    [ 0.10, -0.25,  0.05,  1.00,  0.30],
    [ 0.15, -0.40, -0.10,  0.30,  1.00],
])

CONFIDENCE = 0.95
N_SIMS     = 200_000
SEED       = 123

# Monte Carlo Simulation

def run():
    assets     = list(PORTFOLIO.keys())
    notionals  = np.array([v[0] for v in PORTFOLIO.values()])
    ann_vols   = np.array([v[1] for v in PORTFOLIO.values()])
    ann_drifts = np.array([v[2] for v in PORTFOLIO.values()])

    # Daily parameters
    daily_vol   = ann_vols   / np.sqrt(252)
    daily_drift = ann_drifts / 252

    # Correlated shocks via Cholesky decomposition
    cov    = np.diag(daily_vol) @ CORR @ np.diag(daily_vol)
    rng    = np.random.default_rng(SEED)
    shocks = rng.multivariate_normal(daily_drift, cov, size=N_SIMS)  # (N, 5)

    # Dollar P&L per simulation
    pnl_per_asset = shocks * notionals          # (N, 5)
    portfolio_pnl = pnl_per_asset.sum(axis=1)  # (N,)

    # VaR and ES
    var = np.quantile(portfolio_pnl, 1 - CONFIDENCE)
    es  = portfolio_pnl[portfolio_pnl <= var].mean()

    # Print Results
    total_notional = notionals.sum()
    print("=" * 52)
    print("  Monte Carlo Expected Shortfall")
    print("=" * 52)
    print(f"\n  Simulations     : {N_SIMS:,}")
    print(f"  Confidence      : {CONFIDENCE*100:.0f}%")
    print(f"  Total Notional  : ${total_notional:,.0f}\n")

    print("  Portfolio Holdings")
    print("  " + "-" * 38)
    for asset, (notional, vol, _) in PORTFOLIO.items():
        print(f"  {asset:<18} ${notional:>12,.0f}   vol {vol*100:.0f}%")

    print(f"\n  {'VaR  (95%)':.<30} ${var:>12,.0f}")
    print(f"  {'ES   (95%)':.<30} ${es:>12,.0f}")
    print(f"  {'ES as % of Notional':.<30} {es/total_notional*100:>11.3f}%")
    print("\n" + "=" * 52)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    ax.hist(portfolio_pnl / 1000, bins=120, density=True,
            color="#1e3a5f", edgecolor="none", alpha=0.9)

    # Shade ES region
    tail_vals = portfolio_pnl[portfolio_pnl <= var] / 1000
    ax.hist(tail_vals, bins=60, density=True,
            color="r", edgecolor="none", alpha=0.85, label="Tail (ES region)")

    ax.axvline(var / 1000, color="y", lw=1.8, ls="--",
               label=f"VaR 95% = ${var/1000:,.1f}K")
    ax.axvline(es  / 1000, color="r", lw=2.0, ls="-",
               label=f"ES  95% = ${es/1000:,.1f}K")

    ax.set_title("Portfolio Daily P&L Distribution  |  Monte Carlo ES",
                 color="white", fontsize=13, pad=14)
    ax.set_xlabel("Daily P&L ($000s)", color="silver", fontsize=11)
    ax.set_ylabel("Density",           color="silver", fontsize=11)
    ax.tick_params(colors="silver")
    for spine in ax.spines.values():
        spine.set_edgecolor("k")

    ax.legend(fontsize=10, labelcolor="white",
              facecolor="k", edgecolor="gray")

    plt.tight_layout()
    plt.savefig("es_montecarlo.png",
                dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print("\n  Chart saved: es_montecarlo.png")

if __name__ == "__main__":
    run()