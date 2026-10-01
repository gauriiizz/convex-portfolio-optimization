import matplotlib.pyplot as plt
import pandas as pd
from src.data import fetch_historical_prices, compute_daily_returns
from src.estimators import compute_sample_covariance, compute_ledoit_wolf_covariance, inspect_spectrum
from src.backtest import run_rolling_backtest, compute_performance_metrics

def main():
    print("==========================================================")
    print(" Convex Portfolio Optimization: Spectral Regularization   ")
    print("==========================================================")
    
    print("\n[1/4] Fetching historical market data for Nifty 20 universe...")
    prices = fetch_historical_prices()
    returns = compute_daily_returns(prices)
    n_days, n_assets = returns.shape
    print(f"Loaded {n_days} daily trading rows across {n_assets} assets.")

    print("\n[2/4] Stress-testing Covariance Spectra (N=20, T=60 window)...")
    stressed_window = returns.iloc[-60:].values
    S = compute_sample_covariance(stressed_window)
    Sigma_shrunk, delta = compute_ledoit_wolf_covariance(stressed_window)

    spec_S = inspect_spectrum(S)
    spec_shrunk = inspect_spectrum(Sigma_shrunk)

    print(f"Optimal Ledoit-Wolf Shrinkage Intensity (delta*): {delta:.4f}")
    print(f"Sample Covariance Condition Number       (S):     {spec_S['condition_number']:.2f}")
    print(f"Shrunk Covariance Condition Number (Sigma*):     {spec_shrunk['condition_number']:.2f}")
    print(f"Spectral Floor Elevation (lambda_min): {spec_S['lambda_min']:.6f} -> {spec_shrunk['lambda_min']:.6f}")

    print("\n[3/4] Running rolling out-of-sample backtest with friction...")
    results = run_rolling_backtest(
        returns,
        lookback_window=90,
        rebalance_freq=21,
        gamma=2.5,
        max_weight=0.15,
        transaction_cost_bps=10.0
    )

    metrics_sample = compute_performance_metrics(results["sample"])
    metrics_shrunk = compute_performance_metrics(results["shrunk"])

    summary_df = pd.DataFrame([metrics_sample, metrics_shrunk], index=["Sample Covariance", "Ledoit-Wolf Shrunk"])
    print("\n[4/4] Out-of-Sample Performance Summary:")
    print(summary_df.to_string())

    cum_sample = (1.0 + results["sample"]).cumprod()
    cum_shrunk = (1.0 + results["shrunk"]).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(cum_sample.index, cum_sample.values, label=f"Sample Covariance (Sharpe: {metrics_sample['Sharpe Ratio']})", color="crimson", lw=1.5)
    plt.plot(cum_shrunk.index, cum_shrunk.values, label=f"Ledoit-Wolf Shrunk (Sharpe: {metrics_shrunk['Sharpe Ratio']})", color="navy", lw=2.0)
    plt.title("Out-of-Sample Equity Growth: Sample vs. Shrunk Covariance (Nifty 20)", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Growth of Re. 1.00")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("out_of_sample_performance.png", dpi=300)
    print("\nSaved performance plot as 'out_of_sample_performance.png'. Execution complete.")

if __name__ == "__main__":
    main()
