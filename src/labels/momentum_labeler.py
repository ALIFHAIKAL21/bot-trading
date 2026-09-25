"""
Momentum Event Extractor & Multi-RR Labeler (Point 3: RR 1:1 & RR 1:2)
Transforms continuous time series into discrete transaction setups:
Model learns from historical transaction outcomes (Win vs Loss) at 1:1 and 1:2 RR.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def extract_momentum_setups(
    df: pd.DataFrame,
    max_holding_bars: int = 48,
    atr_sl_mult: float = 1.5,
    min_sl_atr: float = 0.5,
    max_sl_atr: float = 2.5,
) -> pd.DataFrame:
    """
    Extracts all candidate momentum setups from indicator & market structure triggers,
    and calculates actual transaction outcomes for RR 1:1 and RR 1:2.
    
    Args:
        df: DataFrame containing price, indicators, and market structure columns.
        max_holding_bars: Horizon to resolve trade (e.g. 48 bars).
        atr_sl_mult: Fallback SL distance in ATR units.
        min_sl_atr: Minimum acceptable SL distance in ATR.
        max_sl_atr: Maximum acceptable SL distance in ATR.
        
    Returns:
        DataFrame where each row is a historical transaction setup with:
        - Trigger metadata (timestamp, trigger type, direction, entry, SL, TP1, TP2)
        - Outcome labels (result_1r, result_2r, bars_held, mfe_r, mae_r)
        - All feature state at entry bar
    """
    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr_14"].values if "atr_14" in df.columns else np.ones(n)
    
    # Check trigger availability
    has_sma = "sma_cross_bull" in df.columns
    has_smi = "smi_os_reversal_bull" in df.columns
    has_news = "is_news_blackout" in df.columns
    has_struct = "recent_swing_low" in df.columns
    
    setups = []
    
    for i in range(25, n - max_holding_bars):
        # 1. News filter (Point 5): Stop total trade during news blackout
        if has_news and df["is_news_blackout"].iloc[i]:
            continue
            
        c_price = closes[i]
        c_atr = atrs[i] if not np.isnan(atrs[i]) and atrs[i] > 1e-4 else 1.0
        
        # Check potential triggers
        bull_triggers = []
        bear_triggers = []
        
        if has_sma:
            if df["sma_cross_bull"].iloc[i] == 1:
                bull_triggers.append("sma_cross")
            if df["sma_cross_bear"].iloc[i] == 1:
                bear_triggers.append("sma_cross")
                
        if has_smi:
            if df["smi_os_reversal_bull"].iloc[i] == 1:
                bull_triggers.append("smi_os_reversal")
            if df["smi_ob_reversal_bear"].iloc[i] == 1:
                bear_triggers.append("smi_ob_reversal")
            if df["smi_zero_cross_bull"].iloc[i] == 1:
                bull_triggers.append("smi_zero_cross")
            if df["smi_zero_cross_bear"].iloc[i] == 1:
                bear_triggers.append("smi_zero_cross")
                
        # Candidate directions
        directions = []
        if bull_triggers:
            directions.append(("BUY", bull_triggers))
        if bear_triggers:
            directions.append(("SELL", bear_triggers))
            
        for direction, trig_list in directions:
            entry_price = c_price
            
            # 2. Determine Stop Loss (SL) and 1R distance
            if direction == "BUY":
                # Ideal SL is recent confirmed swing low
                sl_cand = df["recent_swing_low"].iloc[i] if has_struct else np.nan
                dist_cand = (entry_price - sl_cand) / c_atr if not np.isnan(sl_cand) else np.nan
                
                if not np.isnan(dist_cand) and min_sl_atr <= dist_cand <= max_sl_atr:
                    sl_price = sl_cand
                else:
                    sl_price = entry_price - (atr_sl_mult * c_atr)
                    
                risk_1r = entry_price - sl_price
                if risk_1r <= 0:
                    continue
                    
                tp1_price = entry_price + 1.0 * risk_1r  # RR 1:1
                tp2_price = entry_price + 2.0 * risk_1r  # RR 1:2
                
            else:  # SELL
                sl_cand = df["recent_swing_high"].iloc[i] if has_struct else np.nan
                dist_cand = (sl_cand - entry_price) / c_atr if not np.isnan(sl_cand) else np.nan
                
                if not np.isnan(dist_cand) and min_sl_atr <= dist_cand <= max_sl_atr:
                    sl_price = sl_cand
                else:
                    sl_price = entry_price + (atr_sl_mult * c_atr)
                    
                risk_1r = sl_price - entry_price
                if risk_1r <= 0:
                    continue
                    
                tp1_price = entry_price - 1.0 * risk_1r  # RR 1:1
                tp2_price = entry_price - 2.0 * risk_1r  # RR 1:2
                
            # 3. Simulate Forward Path (Walk Forward Labeling)
            result_1r = 0   # +1 Win, -1 Loss, 0 Timeout
            result_2r = 0   # +1 Win, -1 Loss, 0 Timeout
            bars_held_1r = max_holding_bars
            bars_held_2r = max_holding_bars
            resolved_1r = False
            resolved_2r = False
            
            max_fav_move = 0.0
            max_adv_move = 0.0
            
            for step in range(1, max_holding_bars + 1):
                idx = i + step
                f_high = highs[idx]
                f_low = lows[idx]
                
                if direction == "BUY":
                    fav = f_high - entry_price
                    adv = entry_price - f_low
                    max_fav_move = max(max_fav_move, fav)
                    max_adv_move = max(max_adv_move, adv)
                    
                    # Check SL hit
                    sl_hit = f_low <= sl_price
                    # Check TP1 hit
                    tp1_hit = f_high >= tp1_price
                    # Check TP2 hit
                    tp2_hit = f_high >= tp2_price
                    
                    # 1:1 RR Resolution
                    if not resolved_1r:
                        if sl_hit and tp1_hit:
                            # Intrabar collision -> Conservative: assume SL
                            result_1r = -1
                            bars_held_1r = step
                            resolved_1r = True
                        elif tp1_hit:
                            result_1r = 1
                            bars_held_1r = step
                            resolved_1r = True
                        elif sl_hit:
                            result_1r = -1
                            bars_held_1r = step
                            resolved_1r = True
                            
                    # 1:2 RR Resolution
                    if not resolved_2r:
                        if sl_hit and tp2_hit:
                            result_2r = -1
                            bars_held_2r = step
                            resolved_2r = True
                        elif tp2_hit:
                            result_2r = 1
                            bars_held_2r = step
                            resolved_2r = True
                        elif sl_hit:
                            result_2r = -1
                            bars_held_2r = step
                            resolved_2r = True
                            
                else:  # SELL
                    fav = entry_price - f_low
                    adv = f_high - entry_price
                    max_fav_move = max(max_fav_move, fav)
                    max_adv_move = max(max_adv_move, adv)
                    
                    sl_hit = f_high >= sl_price
                    tp1_hit = f_low <= tp1_price
                    tp2_hit = f_low <= tp2_price
                    
                    # 1:1 RR Resolution
                    if not resolved_1r:
                        if sl_hit and tp1_hit:
                            result_1r = -1
                            bars_held_1r = step
                            resolved_1r = True
                        elif tp1_hit:
                            result_1r = 1
                            bars_held_1r = step
                            resolved_1r = True
                        elif sl_hit:
                            result_1r = -1
                            bars_held_1r = step
                            resolved_1r = True
                            
                    # 1:2 RR Resolution
                    if not resolved_2r:
                        if sl_hit and tp2_hit:
                            result_2r = -1
                            bars_held_2r = step
                            resolved_2r = True
                        elif tp2_hit:
                            result_2r = 1
                            bars_held_2r = step
                            resolved_2r = True
                        elif sl_hit:
                            result_2r = -1
                            bars_held_2r = step
                            resolved_2r = True
                            
                if resolved_1r and resolved_2r:
                    break
                    
            # Record setup row
            entry_row = df.iloc[i].to_dict()
            entry_row.update({
                "setup_bar_idx": i,
                "direction": direction,
                "direction_num": 1 if direction == "BUY" else -1,
                "primary_trigger": trig_list[0],
                "all_triggers": ",".join(trig_list),
                "trigger_count": len(trig_list),
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp1_price": tp1_price,
                "tp2_price": tp2_price,
                "risk_1r": risk_1r,
                "risk_atr_ratio": risk_1r / c_atr,
                "result_1r": result_1r,
                "result_2r": result_2r,
                "win_1r": 1 if result_1r == 1 else 0,
                "win_2r": 1 if result_2r == 1 else 0,
                "bars_held_1r": bars_held_1r,
                "bars_held_2r": bars_held_2r,
                "mfe_r": max_fav_move / risk_1r,
                "mae_r": max_adv_move / risk_1r,
            })
            setups.append(entry_row)
            
    res_df = pd.DataFrame(setups)
    logger.info(f"Extracted {len(res_df)} momentum transaction setups.")
    return res_df
