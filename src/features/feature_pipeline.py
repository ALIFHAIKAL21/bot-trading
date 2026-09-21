"""Strictly causal feature engineering pipeline.

Guarantees zero look-ahead bias:
- All indicators computed on past data only.
- Strict whitelist enforcement: features fed to any model must match explicit config whitelist.
- Loud LeakageViolationError if any column name matches label/weight/target/forward patterns.
- Higher-timeframe features (4h, 1d) shifted by 1 full bar before forward-filling.
- Zero-volume and gap bars excluded from baseline volume statistics.
"""

import re
from typing import List, Optional, Set
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import FeatureConfig


class LeakageViolationError(ValueError):
    """Raised when a label, weight, target, or forward-looking column is detected in features."""
    pass


FORBIDDEN_LEAKAGE_REGEX = re.compile(r"(label|weight|target|fwd|forward)", re.IGNORECASE)


def validate_feature_columns(cols: List[str], whitelist: Optional[List[str]] = None) -> None:
    """Validate that feature columns contain zero target/weight leakage and match whitelist."""
    for col in cols:
        if FORBIDDEN_LEAKAGE_REGEX.search(col):
            raise LeakageViolationError(
                f"CRITICAL LEAKAGE DETECTED: Feature column '{col}' matches forbidden pattern "
                "'(label|weight|target|fwd|forward)'. It is strictly prohibited from model inputs."
            )

    if whitelist is not None:
        extra_cols = set(cols) - set(whitelist)
        if extra_cols:
            raise LeakageViolationError(
                f"FEATURE WHITELIST VIOLATION: Columns {extra_cols} are not in the explicit config whitelist!"
            )


class FeaturePipeline:
    """Computes technical, cyclical, and multi-timeframe features causally."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute full feature matrix from OHLCV dataframe."""
        if df.empty:
            return pd.DataFrame()

        df_feat = df.copy()
        c = df_feat["close"]
        h = df_feat["high"]
        l = df_feat["low"]
        o = df_feat["open"]
        v = df_feat["volume"]

        # 1. Log Returns
        log_c = np.log(c)
        for w in self.config.return_windows:
            df_feat[f"log_ret_{w}"] = log_c - log_c.shift(w)

        # 2. Realized Volatility (rolling std of 1h log returns)
        log_ret_1 = df_feat["log_ret_1"]
        for w in self.config.volatility_windows:
            df_feat[f"realized_vol_{w}"] = log_ret_1.rolling(w).std()

        # 3. Average True Range (ATR) & Normalized ATR
        prev_c = c.shift(1)
        tr1 = h - l
        tr2 = (h - prev_c).abs()
        tr3 = (l - prev_c).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=self.config.atr_period, adjust=False).mean()
        df_feat["atr_14"] = atr
        df_feat["natr_14"] = atr / c

        # 4. Relative Strength Index (RSI)
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / self.config.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.config.rsi_period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df_feat["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # 5. MACD
        fast_ema = c.ewm(span=self.config.macd_fast, adjust=False).mean()
        slow_ema = c.ewm(span=self.config.macd_slow, adjust=False).mean()
        macd = (fast_ema - slow_ema) / c
        macd_signal = macd.ewm(span=self.config.macd_signal, adjust=False).mean()
        df_feat["macd"] = macd
        df_feat["macd_signal"] = macd_signal
        df_feat["macd_hist"] = macd - macd_signal

        # 6. Bollinger Bands
        bb_mid = c.rolling(self.config.bollinger_period).mean()
        bb_std = c.rolling(self.config.bollinger_period).std()
        bb_upper = bb_mid + self.config.bollinger_std * bb_std
        bb_lower = bb_mid - self.config.bollinger_std * bb_std
        df_feat["boll_pct_b"] = (c - bb_lower) / ((bb_upper - bb_lower) + 1e-9)
        df_feat["boll_bandwidth"] = (bb_upper - bb_lower) / bb_mid

        # 7. Volume Z-Score (excluding zero-volume/gap bars from baseline distribution)
        valid_vol = v.replace(0.0, np.nan)
        vol_mean = valid_vol.rolling(self.config.volume_zscore_window, min_periods=4).mean()
        vol_std = valid_vol.rolling(self.config.volume_zscore_window, min_periods=4).std()
        df_feat["vol_zscore_24"] = ((v - vol_mean) / (vol_std + 1e-9)).fillna(0.0)

        # 8. Candle Anatomy Ratios
        candle_range = (h - l).replace(0, np.nan)
        df_feat["body_ratio"] = (c - o).abs() / candle_range
        df_feat["upper_wick_ratio"] = (h - np.maximum(c, o)) / candle_range
        df_feat["lower_wick_ratio"] = (np.minimum(c, o) - l) / candle_range
        df_feat["body_ratio"] = df_feat["body_ratio"].fillna(0.0)
        df_feat["upper_wick_ratio"] = df_feat["upper_wick_ratio"].fillna(0.0)
        df_feat["lower_wick_ratio"] = df_feat["lower_wick_ratio"].fillna(0.0)

        # 9. Rolling Skewness and Kurtosis of Returns
        for w in self.config.skew_kurtosis_windows:
            df_feat[f"skew_{w}"] = log_ret_1.rolling(w).skew()
            df_feat[f"kurt_{w}"] = log_ret_1.rolling(w).kurt()

        # 10. Higher-Timeframe (HTF) Trend Features - Strictly Causal
        # Resample to completed HTF bar, calculate EMA trend, shift by 1 full HTF bar, ffill to 1h
        for tf in self.config.htf_timeframes:
            rule = "4h" if tf == "4h" else "1D"
            htf_df = pd.DataFrame(
                {
                    "open": o.resample(rule).first(),
                    "high": h.resample(rule).max(),
                    "low": l.resample(rule).min(),
                    "close": c.resample(rule).last(),
                }
            ).dropna()

            htf_ema = htf_df["close"].ewm(span=20, adjust=False).mean()
            htf_trend = (htf_df["close"] - htf_ema) / (htf_ema + 1e-9)

            # Shift by 1 HTF bar so bar t only sees completed past HTF bars
            htf_shifted = htf_trend.shift(1)

            # Reindex to 1h index and forward fill
            df_feat[f"htf_{tf}_trend"] = htf_shifted.reindex(df.index, method="ffill")

        # 11. Cyclical Calendar Encodings
        timestamps = df.index
        hours = timestamps.hour
        dows = timestamps.dayofweek
        df_feat["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
        df_feat["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
        df_feat["sin_dow"] = np.sin(2 * np.pi * dows / 7.0)
        df_feat["cos_dow"] = np.cos(2 * np.pi * dows / 7.0)

        return df_feat

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return validated list of feature column names strictly adhering to whitelist."""
        # Find all whitelist columns present in df
        present_cols = [col for col in self.config.whitelist if col in df.columns]
        # Validate against leakage regex and whitelist
        validate_feature_columns(present_cols, self.config.whitelist)
        return present_cols
