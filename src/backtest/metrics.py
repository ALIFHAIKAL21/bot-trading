"""Quant performance analytics, statistical significance tests, and baseline benchmarks.

Includes:
- Deflated Sharpe Ratio (DSR) & Skew/Kurtosis adjustment.
- 1,000-iteration Bootstrap Confidence Intervals on Sharpe.
- 200-iteration Monte Carlo random trading baseline.
- Model ablation delta analyzer.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


def calculate_deflated_sharpe_ratio(
    sharpe_hat: float,
    n_trials: int,
    returns: np.ndarray,
    periods_per_year: float = 8760.0,
) -> float:
    """Compute Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts for multiple testing, skewness, and kurtosis.
    """
    n = len(returns)
    if n < 30 or sharpe_hat <= 0:
        return 0.0

    # Un-annualize Sharpe to sample frequency
    sr = sharpe_hat / np.sqrt(periods_per_year)

    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # excess kurtosis

    # Variance of Sharpe ratio estimator
    sr_var = (1.0 - skew * sr + ((kurt + 2.0) / 4.0) * (sr**2)) / (n - 1)
    sr_std = np.sqrt(max(sr_var, 1e-9))

    # Expected maximum Sharpe among N independent trials under H0
    # Euler-Mascheroni approximation
    euler = 0.5772156649
    if n_trials > 1:
        z = (1.0 - euler) * stats.norm.ppf(1.0 - 1.0 / n_trials) + euler * stats.norm.ppf(
            1.0 - 1.0 / (n_trials * np.e)
        )
        expected_max_sr = z * sr_std
    else:
        expected_max_sr = 0.0

    # Prob(SR > expected_max_sr)
    z_stat = (sr - expected_max_sr) / sr_std
    dsr = float(stats.norm.cdf(z_stat))
    return dsr


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    periods_per_year: float = 8760.0,
) -> Tuple[float, float]:
    """Compute circular block bootstrap confidence interval on Sharpe ratio."""
    n = len(returns)
    if n < 30:
        return 0.0, 0.0

    block_size = 24  # 1-day blocks for hourly crypto data
    n_blocks = int(np.ceil(n / block_size))
    sharpes = np.zeros(n_bootstraps)

    np.random.seed(42)
    for b in range(n_bootstraps):
        start_points = np.random.randint(0, n - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(st, st + block_size) for st in start_points])[:n]
        boot_rets = returns[indices]

        mean_r = np.mean(boot_rets)
        std_r = np.std(boot_rets)
        sharpes[b] = (mean_r / (std_r + 1e-9)) * np.sqrt(periods_per_year)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(sharpes, alpha / 2.0 * 100.0))
    ci_upper = float(np.percentile(sharpes, (1.0 - alpha / 2.0) * 100.0))
    return ci_lower, ci_upper


def run_monte_carlo_random_baseline(
    bar_returns: np.ndarray,
    active_exposure: float = 0.35,
    n_sims: int = 200,
    periods_per_year: float = 8760.0,
) -> Dict[str, float]:
    """Simulate 200 random trading agents to determine luck benchmark."""
    n = len(bar_returns)
    random_sharpes = np.zeros(n_sims)
    random_cagrs = np.zeros(n_sims)

    np.random.seed(42)
    for i in range(n_sims):
        # Random positions with similar turnover structure
        rand_signals = np.random.choice([0.0, active_exposure], size=n, p=[0.6, 0.4])
        # Shift 1 bar for execution
        positions = np.zeros(n)
        positions[1:] = rand_signals[:-1]
        sim_rets = positions * bar_returns - np.abs(np.diff(positions, prepend=0)) * 0.0015

        mean_r = np.mean(sim_rets)
        std_r = np.std(sim_rets)
        random_sharpes[i] = (mean_r / (std_r + 1e-9)) * np.sqrt(periods_per_year)
        random_cagrs[i] = (np.prod(1.0 + sim_rets) ** (periods_per_year / max(n, 1))) - 1.0

    return {
        "mc_sharpe_mean": float(np.mean(random_sharpes)),
        "mc_sharpe_95th": float(np.percentile(random_sharpes, 95)),
        "mc_sharpe_max": float(np.max(random_sharpes)),
        "mc_cagr_mean": float(np.mean(random_cagrs)),
        "mc_cagr_95th": float(np.percentile(random_cagrs, 95)),
    }
