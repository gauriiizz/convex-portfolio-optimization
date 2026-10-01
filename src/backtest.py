from typing import Dict
import numpy as np
import pandas as pd
from .estimators import compute_sample_covariance, compute_ledoit_wolf_covariance
from .optimizer import solve_markowitz_qp

def run_rolling_backtest(
    returns_df: pd.DataFrame,
    lookback_window: int = 90,
    rebalance_freq: int = 21,
    gamma: float = 2.0,
    max_weight: float = 0.15,
    transaction_cost_bps: float = 10.0
) -> Dict[str, pd.Series]:
    n_days, n_assets = returns_df.shape
    dates = returns_df.index
    
    sample_port_returns = []
    shrunk_port_returns = []
    eval_dates = []
    
    w_sample_prev = np.ones(n_assets) / n_assets
    w_shrunk_prev = np.ones(n_assets) / n_assets
    cost_factor = transaction_cost_bps / 10000.0

    for t in range(lookback_window, n_days - rebalance_freq, rebalance_freq):
        window_returns = returns_df.iloc[t - lookback_window:t].values
        mu = np.mean(window_returns, axis=0) * 252
        
        S = compute_sample_covariance(window_returns) * 252
        Sigma_shrunk, _ = compute_ledoit_wolf_covariance(window_returns)
        Sigma_shrunk = Sigma_shrunk * 252
        
        try:
            w_sample = solve_markowitz_qp(mu, S, gamma=gamma, max_weight=max_weight)
            w_shrunk = solve_markowitz_qp(mu, Sigma_shrunk, gamma=gamma, max_weight=max_weight)
        except Exception:
            w_sample, w_shrunk = w_sample_prev, w_shrunk_prev

        cost_sample = cost_factor * np.sum(np.abs(w_sample - w_sample_prev))
        cost_shrunk = cost_factor * np.sum(np.abs(w_shrunk - w_shrunk_prev))
        w_sample_prev, w_shrunk_prev = w_sample, w_shrunk

        next_period_returns = returns_df.iloc[t:t + rebalance_freq].values
        period_dates = dates[t:t + rebalance_freq]

        for i, daily_ret in enumerate(next_period_returns):
            r_sample = float(np.dot(w_sample, daily_ret))
            r_shrunk = float(np.dot(w_shrunk, daily_ret))
            if i == 0:
                r_sample -= cost_sample
                r_shrunk -= cost_shrunk
            sample_port_returns.append(r_sample)
            shrunk_port_returns.append(r_shrunk)
            eval_dates.append(period_dates[i])

    return {
        "sample": pd.Series(sample_port_returns, index=eval_dates),
        "shrunk": pd.Series(shrunk_port_returns, index=eval_dates)
    }

def compute_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    ann_return = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min())
    
    return {
        "Annualized Return (%)": round(ann_return * 100, 2),
        "Annualized Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd * 100, 2)
    }
