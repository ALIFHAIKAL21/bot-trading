"""
Indicator Signals: MA Cross and Stochastic Momentum Index (SMI)
Matches TradingView built-in indicator standards exactly.
"""

import numpy as np
import pandas as pd


def compute_ma_cross(
    df: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21,
    prefix: str = "sma",
) -> pd.DataFrame:
    """
    Computes Moving Average Cross signals matching TradingView 'MA Cross' indicator.
    
    Args:
        df: DataFrame with 'close' column
        fast_period: Period for fast MA (TradingView default = 9)
        slow_period: Period for slow MA (TradingView default = 21)
        prefix: 'sma' or 'ema'
        
    Returns:
        DataFrame with MA columns and cross signals
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    
    # Calculate MAs
    if prefix == "sma":
        fast_ma = close.rolling(window=fast_period, min_periods=fast_period).mean()
        slow_ma = close.rolling(window=slow_period, min_periods=slow_period).mean()
    else:
        fast_ma = close.ewm(span=fast_period, adjust=False).mean()
        slow_ma = close.ewm(span=slow_period, adjust=False).mean()
        
    out[f"{prefix}_{fast_period}"] = fast_ma
    out[f"{prefix}_{slow_period}"] = slow_ma
    
    # Trend direction: 1 for bullish, -1 for bearish
    trend = np.where(fast_ma > slow_ma, 1, np.where(fast_ma < slow_ma, -1, 0))
    out[f"{prefix}_trend"] = trend
    
    # Normalized spread
    out[f"{prefix}_spread"] = (fast_ma - slow_ma) / (close + 1e-9)
    
    # Cross detection (causal, on bar close)
    prev_fast = fast_ma.shift(1)
    prev_slow = slow_ma.shift(1)
    
    # Bullish Cross: fast was <= slow, now fast > slow
    bull_cross = (prev_fast <= prev_slow) & (fast_ma > slow_ma)
    # Bearish Cross: fast was >= slow, now fast < slow
    bear_cross = (prev_fast >= prev_slow) & (fast_ma < slow_ma)
    
    out[f"{prefix}_cross_bull"] = bull_cross.astype(int)
    out[f"{prefix}_cross_bear"] = bear_cross.astype(int)
    out[f"{prefix}_cross_signal"] = np.where(bull_cross, 1, np.where(bear_cross, -1, 0))
    
    # Bars since last cross
    cross_occurred = bull_cross | bear_cross
    bars_since = np.zeros(len(df), dtype=int)
    count = 999  # large initial count
    for i in range(len(df)):
        if cross_occurred.iloc[i]:
            count = 0
        else:
            count += 1
        bars_since[i] = count
    out[f"{prefix}_bars_since_cross"] = bars_since
    
    return out


def compute_smi(
    df: pd.DataFrame,
    q: int = 10,
    r: int = 3,
    s: int = 3,
    d: int = 10,
    ob_level: float = 40.0,
    os_level: float = -40.0,
) -> pd.DataFrame:
    """
    Computes Stochastic Momentum Index (SMI) matching TradingView built-in indicator.
    
    TradingView Formula:
        hh = highest(high, q)
        ll = lowest(low, q)
        diff = close - 0.5 * (hh + ll)
        rdiff = hh - ll
        diff_smoothed = ema(ema(diff, r), s)
        rdiff_smoothed = ema(ema(rdiff, r), s)
        smi = 100 * (diff_smoothed / (0.5 * rdiff_smoothed))
        signal = ema(smi, d)
        
    Args:
        df: DataFrame with 'high', 'low', 'close'
        q: %K Length (lookback, TradingView default = 10)
        r: %K Smoothing (TradingView default = 3)
        s: %K Double Smoothing (TradingView default = 3)
        d: %D Length (Signal period, TradingView default = 10)
        ob_level: Overbought threshold (default = +40.0)
        os_level: Oversold threshold (default = -40.0)
        
    Returns:
        DataFrame with SMI columns and signals
    """
    out = pd.DataFrame(index=df.index)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # Range & mid
    hh = high.rolling(window=q, min_periods=q).max()
    ll = low.rolling(window=q, min_periods=q).min()
    
    diff = close - 0.5 * (hh + ll)
    rdiff = hh - ll
    
    # Double exponential smoothing
    diff_s1 = diff.ewm(span=r, adjust=False).mean()
    diff_s2 = diff_s1.ewm(span=s, adjust=False).mean()
    
    rdiff_s1 = rdiff.ewm(span=r, adjust=False).mean()
    rdiff_s2 = rdiff_s1.ewm(span=s, adjust=False).mean()
    
    half_rdiff = 0.5 * rdiff_s2
    # Protect against divide-by-zero
    smi = np.where(half_rdiff > 1e-9, 100.0 * (diff_s2 / half_rdiff), 0.0)
    smi_series = pd.Series(smi, index=df.index)
    
    # Signal line is EMA of SMI
    signal_series = smi_series.ewm(span=d, adjust=False).mean()
    
    out["smi_val"] = smi_series
    out["smi_signal"] = signal_series
    out["smi_hist"] = smi_series - signal_series
    
    # Zone indicators (-1 = Oversold < -40, +1 = Overbought > +40, 0 = Neutral)
    zone = np.where(smi_series > ob_level, 1, np.where(smi_series < os_level, -1, 0))
    out["smi_zone"] = zone
    
    # Crossovers between SMI line and Signal line
    prev_smi = smi_series.shift(1)
    prev_sig = signal_series.shift(1)
    
    smi_cross_bull = (prev_smi <= prev_sig) & (smi_series > signal_series)
    smi_cross_bear = (prev_smi >= prev_sig) & (smi_series < signal_series)
    
    out["smi_cross_bull"] = smi_cross_bull.astype(int)
    out["smi_cross_bear"] = bear_cross = smi_cross_bear.astype(int)
    
    # High-quality momentum triggers:
    # 1. Bullish cross occurring inside or emerging from Oversold (< -40)
    out["smi_os_reversal_bull"] = (
        smi_cross_bull & ((prev_smi < os_level) | (smi_series < os_level + 10.0))
    ).astype(int)
    
    # 2. Bearish cross occurring inside or emerging from Overbought (> +40)
    out["smi_ob_reversal_bear"] = (
        smi_cross_bear & ((prev_smi > ob_level) | (smi_series > ob_level - 10.0))
    ).astype(int)
    
    # 3. Midline (Zero) cross
    out["smi_zero_cross_bull"] = ((prev_smi <= 0.0) & (smi_series > 0.0)).astype(int)
    out["smi_zero_cross_bear"] = ((prev_smi >= 0.0) & (smi_series < 0.0)).astype(int)
    
    return out


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes both MA Cross (SMA 9/21, EMA 9/21) and SMI (10, 3, 3, 10).
    Appends all resulting columns to df.
    """
    sma_df = compute_ma_cross(df, fast_period=9, slow_period=21, prefix="sma")
    ema_df = compute_ma_cross(df, fast_period=9, slow_period=21, prefix="ema")
    smi_df = compute_smi(df, q=10, r=3, s=3, d=10, ob_level=40.0, os_level=-40.0)
    
    return pd.concat([df, sma_df, ema_df, smi_df], axis=1)
