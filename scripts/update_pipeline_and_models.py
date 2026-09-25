"""
Script to update feature_config.py, feature_pipeline.py, dataset.py, and moment_model.py
for the 15-channel non-destructive upgrade.
"""
import pathlib

# 1. Update feature_config.py
cfg_path = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\src\pipeline\feature_config.py')
cfg_text = cfg_path.read_text(encoding='utf-8')

new_channels_str = '''FEATURE_CHANNELS: Dict[int, str] = {
    0: "ohlc_norm_open",      # Channel 1: Open normalized (EMA/ATR)
    1: "ohlc_norm_high",      # Channel 2: High normalized
    2: "ohlc_norm_low",       # Channel 3: Low normalized
    3: "ohlc_norm_close",     # Channel 4: Close normalized
    4: "volume_zscore",       # Channel 5: Volume dynamic Z-score
    5: "sma_cross_spread",    # Channel 6: MA Cross (SMA 9 x SMA 21 spread / ATR)
    6: "smi_val",             # Channel 7: Stochastic Momentum Index (10, 3, 3, 10) in [-1, 1]
    7: "smi_signal_hist",     # Channel 8: SMI Signal Line Difference / Momentum
    8: "smi_reversal_zone",   # Channel 9: SMI +/-40 Reversal Signal (+1 OS bull, -1 OB bear)
    9: "order_block_zone",    # Channel 10: Supply / Demand Order Block (+1 Demand, -1 Supply)
    10: "liquidity_sweep",    # Channel 11: Liquidity Sweeps (+1 SSL sweep, -1 BSL sweep)
    11: "valuation_regime",   # Channel 12: Premium/Discount Valuation & Sideways Regime
    12: "h4_macro_trend_velocity",        # Channel 13: H4 Macro Trend Velocity ([-1, 1])
    13: "h4_market_structure",            # Channel 14: H4 Structural Bull/Bear Status ([-1, 1])
    14: "momentum_expansion_persistence", # Channel 15: Macro Momentum & Expansion Persistence ([-1, 1])
}

NUM_CHANNELS: int = len(FEATURE_CHANNELS)  # = 15'''

# Replace the FEATURE_CHANNELS block
import re
pattern = re.compile(r'FEATURE_CHANNELS: Dict\[int, str\] = \{.*?NUM_CHANNELS: int = len\(FEATURE_CHANNELS\).*?(?=\n\n#|\n#|\Z)', re.DOTALL)
if pattern.search(cfg_text):
    cfg_text = pattern.sub(new_channels_str, cfg_text)
    cfg_path.write_text(cfg_text, encoding='utf-8')
    print("1. feature_config.py updated with 15 channels.")
else:
    print("WARNING: Could not match FEATURE_CHANNELS pattern in feature_config.py")

# 2. Update feature_pipeline.py to import and call build_macro_features
pipe_path = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\src\pipeline\feature_pipeline.py')
pipe_text = pipe_path.read_text(encoding='utf-8')

# Add import if missing
if "from .features_macro import build_macro_features" not in pipe_text:
    pipe_text = pipe_text.replace(
        "from .features_structural import build_structural_features",
        "from .features_structural import build_structural_features\nfrom .features_macro import build_macro_features"
    )

# Add build_macro_features call in build_all_features
old_call = """    # Channel 8-9: Structural features
    if verbose:
        print(f"  [Ch 8-9] Liquidity Distance + FVG Status...")
    df = build_structural_features(df)"""

new_call = """    # Channel 8-9: Structural features
    if verbose:
        print(f"  [Ch 8-9] Liquidity Distance + FVG Status...")
    df = build_structural_features(df)
    
    # Channel 12-14: H4 Macro Multi-Timeframe features (Strictly Causal)
    if verbose:
        print(f"  [Ch 12-14] H4 Macro Velocity + Market Structure + Momentum Expansion...")
    df = build_macro_features(df)"""

if old_call in pipe_text:
    pipe_text = pipe_text.replace(old_call, new_call)
    pipe_path.write_text(pipe_text, encoding='utf-8')
    print("2. feature_pipeline.py updated with macro features call.")
else:
    print("WARNING: Could not find call site in feature_pipeline.py")

# 3. Update dataset.py default channels
ds_path = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\src\models\dataset.py')
ds_text = ds_path.read_text(encoding='utf-8')

old_ds_channels = """DEFAULT_FEATURE_CHANNELS: List[str] = [
    "ohlc_norm_open",      # Ch 1: Open normalized
    "ohlc_norm_high",      # Ch 2: High normalized
    "ohlc_norm_low",       # Ch 3: Low normalized
    "ohlc_norm_close",     # Ch 4: Close normalized
    "volume_zscore",       # Ch 5: Volume Z-score
    "sma_cross_spread",    # Ch 6: MA Cross (SMA 9 x SMA 21 spread / ATR)
    "smi_val",             # Ch 7: Stochastic Momentum Index (10, 3, 3, 10) in [-1, 1]
    "smi_signal_hist",     # Ch 8: SMI Signal Line Difference / Momentum
    "smi_reversal_zone",   # Ch 9: SMI +/-40 Reversal Signal (+1 OS bull, -1 OB bear)
    "order_block_zone",    # Ch 10: Supply / Demand Order Block (+1 Demand, -1 Supply)
    "liquidity_sweep",     # Ch 11: Liquidity Sweeps (+1 SSL sweep, -1 BSL sweep)
    "valuation_regime",    # Ch 12: Premium/Discount Valuation & Sideways Regime
]"""

new_ds_channels = """DEFAULT_FEATURE_CHANNELS: List[str] = [
    "ohlc_norm_open",      # Ch 1: Open normalized
    "ohlc_norm_high",      # Ch 2: High normalized
    "ohlc_norm_low",       # Ch 3: Low normalized
    "ohlc_norm_close",     # Ch 4: Close normalized
    "volume_zscore",       # Ch 5: Volume Z-score
    "sma_cross_spread",    # Ch 6: MA Cross (SMA 9 x SMA 21 spread / ATR)
    "smi_val",             # Ch 7: Stochastic Momentum Index (10, 3, 3, 10) in [-1, 1]
    "smi_signal_hist",     # Ch 8: SMI Signal Line Difference / Momentum
    "smi_reversal_zone",   # Ch 9: SMI +/-40 Reversal Signal (+1 OS bull, -1 OB bear)
    "order_block_zone",    # Ch 10: Supply / Demand Order Block (+1 Demand, -1 Supply)
    "liquidity_sweep",     # Ch 11: Liquidity Sweeps (+1 SSL sweep, -1 BSL sweep)
    "valuation_regime",    # Ch 12: Premium/Discount Valuation & Sideways Regime
    "h4_macro_trend_velocity",        # Ch 13: H4 Macro Velocity
    "h4_market_structure",            # Ch 14: H4 Structural Ribbon Alignment
    "momentum_expansion_persistence", # Ch 15: Expansion Persistence & Proximity
]"""

if old_ds_channels in ds_text:
    ds_text = ds_text.replace(old_ds_channels, new_ds_channels)
    ds_path.write_text(ds_text, encoding='utf-8')
    print("3. dataset.py updated with 15 default channels.")
else:
    print("WARNING: Could not find DEFAULT_FEATURE_CHANNELS in dataset.py")

# 4. Update moment_model.py default n_channels
model_path = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\src\models\moment_model.py')
model_text = model_path.read_text(encoding='utf-8')

old_model_cfg = "n_channels: int = 12"
new_model_cfg = "n_channels: int = 15"
old_model_pe = "n_channels: int = 12,"
new_model_pe = "n_channels: int = 15,"

if old_model_cfg in model_text:
    model_text = model_text.replace(old_model_cfg, new_model_cfg)
    model_text = model_text.replace(old_model_pe, new_model_pe)
    model_path.write_text(model_text, encoding='utf-8')
    print("4. moment_model.py updated with n_channels = 15.")
else:
    print("WARNING: Could not find n_channels in moment_model.py")

print("\nAll pipeline and model specifications successfully upgraded to 15 channels!")
