"""
Market Structure Engine: Point 4 (Supply/Demand, Support/Resistance, Liquidity Sweeps)
Strictly causal implementation: zero lookahead bias.
"""

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Computes Average True Range (ATR) causally."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def compute_market_structure(
    df: pd.DataFrame,
    swing_lookback: int = 3,
    atr_period: int = 14,
    expansion_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Computes SMC / Market Structure features without lookahead bias.
    
    Args:
        df: DataFrame with 'open', 'high', 'low', 'close'
        swing_lookback: Bars required on either side to confirm swing (e.g. 3 bars)
        atr_period: ATR smoothing period
        expansion_mult: Multiplier of ATR to qualify explosive expansion for Supply/Demand
        
    Returns:
        DataFrame with market structure features
    """
    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values
    
    atr = compute_atr(df, period=atr_period).values
    
    # Output arrays
    dist_to_support_atr = np.full(n, np.nan)
    dist_to_resistance_atr = np.full(n, np.nan)
    recent_swing_high = np.full(n, np.nan)
    recent_swing_low = np.full(n, np.nan)
    
    # Liquidity sweeps
    liq_sweep_bull = np.zeros(n, dtype=int)  # Sell-side sweep (SSL) -> Bullish signal
    liq_sweep_bear = np.zeros(n, dtype=int)  # Buy-side sweep (BSL) -> Bearish signal
    
    # Supply & Demand zones
    demand_top = np.full(n, np.nan)
    demand_bot = np.full(n, np.nan)
    supply_top = np.full(n, np.nan)
    supply_bot = np.full(n, np.nan)
    
    dist_to_demand_atr = np.full(n, np.nan)
    dist_to_supply_atr = np.full(n, np.nan)
    inside_demand = np.zeros(n, dtype=int)
    inside_supply = np.zeros(n, dtype=int)
    
    # Active structural memory
    confirmed_swing_highs = []
    confirmed_swing_lows = []
    
    active_demand_zones = []  # list of (bot, top)
    active_supply_zones = []  # list of (bot, top)
    
    k = swing_lookback
    
    for i in range(n):
        c_price = closes[i]
        c_atr = atr[i] if not np.isnan(atr[i]) and atr[i] > 1e-4 else 1.0
        
        # 1. Detect Swing Point confirmed at bar i:
        # Check if candle at (i - k) is higher than k bars before and k bars after (up to i)
        target_idx = i - k
        if target_idx >= k:
            # Check swing high at target_idx
            is_sh = True
            cand_high = highs[target_idx]
            for offset in range(1, k + 1):
                if highs[target_idx - offset] >= cand_high or highs[target_idx + offset] > cand_high:
                    is_sh = False
                    break
            if is_sh:
                confirmed_swing_highs.append(cand_high)
                if len(confirmed_swing_highs) > 10:
                    confirmed_swing_highs.pop(0)
                    
            # Check swing low at target_idx
            is_sl = True
            cand_low = lows[target_idx]
            for offset in range(1, k + 1):
                if lows[target_idx - offset] <= cand_low or lows[target_idx + offset] < cand_low:
                    is_sl = False
                    break
            if is_sl:
                confirmed_swing_lows.append(cand_low)
                if len(confirmed_swing_lows) > 10:
                    confirmed_swing_lows.pop(0)
                    
        # 2. Detect Supply & Demand (Order Blocks) confirmed up to bar i
        # Demand: bearish candle at i-3 followed by 3-bar expansion > 1.5 * ATR
        if i >= 4:
            # Check demand formation ending at i
            exp_move_up = closes[i] - lows[i - 3]
            if exp_move_up > expansion_mult * c_atr and closes[i - 3] < opens[i - 3]:
                # Order block is candle i-3
                active_demand_zones.append((lows[i - 3], highs[i - 3]))
                if len(active_demand_zones) > 5:
                    active_demand_zones.pop(0)
                    
            # Supply: bullish candle at i-3 followed by 3-bar drop > 1.5 * ATR
            exp_move_down = highs[i - 3] - closes[i]
            if exp_move_down > expansion_mult * c_atr and closes[i - 3] > opens[i - 3]:
                active_supply_zones.append((lows[i - 3], highs[i - 3]))
                if len(active_supply_zones) > 5:
                    active_supply_zones.pop(0)
                    
        # 3. Calculate distance to nearest active S/R
        if confirmed_swing_highs:
            # Find nearest resistance above current price
            resists_above = [r for r in confirmed_swing_highs if r >= c_price]
            nearest_res = min(resists_above) if resists_above else confirmed_swing_highs[-1]
            recent_swing_high[i] = confirmed_swing_highs[-1]
            dist_to_resistance_atr[i] = (nearest_res - c_price) / c_atr
            
            # Liquidity Sweep Bear (Buy-side liquidity sweep):
            # High breaks above most recent swing high, but close closes back below it
            last_sh = confirmed_swing_highs[-1]
            if highs[i] > last_sh and closes[i] < last_sh:
                liq_sweep_bear[i] = 1
                
        if confirmed_swing_lows:
            # Find nearest support below current price
            supports_below = [s for s in confirmed_swing_lows if s <= c_price]
            nearest_sup = max(supports_below) if supports_below else confirmed_swing_lows[-1]
            recent_swing_low[i] = confirmed_swing_lows[-1]
            dist_to_support_atr[i] = (c_price - nearest_sup) / c_atr
            
            # Liquidity Sweep Bull (Sell-side liquidity sweep):
            # Low drops below most recent swing low, but close closes back above it
            last_sl = confirmed_swing_lows[-1]
            if lows[i] < last_sl and closes[i] > last_sl:
                liq_sweep_bull[i] = 1
                
        # 4. Supply & Demand distance & inside flag
        if active_demand_zones:
            # Most recent demand zone
            d_bot, d_top = active_demand_zones[-1]
            demand_top[i] = d_top
            demand_bot[i] = d_bot
            if d_bot <= c_price <= d_top:
                inside_demand[i] = 1
                dist_to_demand_atr[i] = 0.0
            else:
                dist_to_demand_atr[i] = (c_price - d_top) / c_atr if c_price > d_top else (d_bot - c_price) / c_atr
                
        if active_supply_zones:
            s_bot, s_top = active_supply_zones[-1]
            supply_top[i] = s_top
            supply_bot[i] = s_bot
            if s_bot <= c_price <= s_top:
                inside_supply[i] = 1
                dist_to_supply_atr[i] = 0.0
            else:
                dist_to_supply_atr[i] = (s_bot - c_price) / c_atr if c_price < s_bot else (c_price - s_top) / c_atr
                
    # 5. Premium / Discount Ratio over 24-bar / 48-bar rolling window
    window = 48
    rolling_high = df["high"].rolling(window=window, min_periods=window // 2).max()
    rolling_low = df["low"].rolling(window=window, min_periods=window // 2).min()
    premium_discount_ratio = (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)
    premium_discount_ratio = premium_discount_ratio.clip(0.0, 1.0)
    
    out = pd.DataFrame(index=df.index)
    out["atr_14"] = atr
    out["dist_to_support_atr"] = dist_to_support_atr
    out["dist_to_resistance_atr"] = dist_to_resistance_atr
    out["recent_swing_high"] = recent_swing_high
    out["recent_swing_low"] = recent_swing_low
    out["liq_sweep_bull"] = liq_sweep_bull
    out["liq_sweep_bear"] = liq_sweep_bear
    out["dist_to_demand_atr"] = dist_to_demand_atr
    out["dist_to_supply_atr"] = dist_to_supply_atr
    out["inside_demand_zone"] = inside_demand
    out["inside_supply_zone"] = inside_supply
    out["premium_discount_ratio"] = premium_discount_ratio
    
    return out
