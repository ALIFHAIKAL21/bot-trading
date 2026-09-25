"""
Script to update feature_config.py, feature_pipeline.py, dataset.py, and moment_model.py
"""
import pathlib

# 1. Update feature_config.py
p_cfg = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\src\pipeline\feature_config.py')
cfg_text = p_cfg.read_text(encoding='utf-8')

old_channels = '''FEATURE_CHANNELS: Dict[int, str] = {
    0: "ohlc_norm_open",      # Channel 1: Open normalized (EMA/ATR)
    1: "ohlc_norm_high",      # Channel 2: High normalized
    2: "ohlc_norm_low",       # Channel 3: Low normalized
    3: "ohlc_norm_close",     # Channel 4: Close normalized
    4: "volume_zscore",       # Channel 5: Volume dynamic Z-score
    5: "smi",                 # Channel 6: Stochastic Momentum Index
    6: "ma_ribbon_slope",     # Channel 7: MA Ribbon slope/differential
    7: "liquidity_distance",  # Channel 8: Distance to nearest liquidity pool
    8: "fvg_status",          # Channel 9: FVG/Order Block status
}'''

new_channels = '''FEATURE_CHANNELS: Dict[int, str] = {
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
}'''

if old_channels in cfg_text:
    cfg_text = cfg_text.replace(old_channels, new_channels)
    p_cfg.write_text(cfg_text, encoding='utf-8')
    print("Updated feature_config.py FEATURE_CHANNELS to 12 channels!")
else:
    print("Warning: old_channels not found in feature_config.py!")

# 2. Update dataset.py
p_dataset = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\src\models\dataset.py')
data_text = p_dataset.read_text(encoding='utf-8')

old_dataset_channels = '''DEFAULT_FEATURE_CHANNELS: List[str] = [
    "ohlc_norm_open",      # Ch 1: Open normalized
    "ohlc_norm_high",      # Ch 2: High normalized
    "ohlc_norm_low",       # Ch 3: Low normalized
    "ohlc_norm_close",     # Ch 4: Close normalized
    "volume_zscore",       # Ch 5: Volume Z-score
    "smi",                 # Ch 6: Stochastic Momentum Index
    "ma_ribbon_slope",     # Ch 7: MA Ribbon Slope
    "liquidity_distance",  # Ch 8: Liquidity Pool Distance
    "fvg_status",          # Ch 9: Fair Value Gap Status
]'''

new_dataset_channels = '''DEFAULT_FEATURE_CHANNELS: List[str] = [
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
]'''

if old_dataset_channels in data_text:
    data_text = data_text.replace(old_dataset_channels, new_dataset_channels)
    p_dataset.write_text(data_text, encoding='utf-8')
    print("Updated dataset.py DEFAULT_FEATURE_CHANNELS to 12 channels!")
else:
    print("Warning: old_dataset_channels not found in dataset.py!")

# 3. Update moment_model.py
p_model = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\src\models\moment_model.py')
model_text = p_model.read_text(encoding='utf-8')

# Update MOMENTConfig n_channels: int = 9 -> 12
if "n_channels: int = 9" in model_text:
    model_text = model_text.replace("n_channels: int = 9", "n_channels: int = 12")
    print("Updated MOMENTConfig n_channels to 12!")

# Update PretrainedMOMENTClassifier __init__ to accept n_channels and pass to PatchEmbedding
old_pretrained_init = '''    def __init__(
        self,
        weights_path: Optional[str] = None,
        num_classes: int = 5,
        dropout: float = 0.2,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
    ):'''

new_pretrained_init = '''    def __init__(
        self,
        weights_path: Optional[str] = None,
        num_classes: int = 5,
        n_channels: int = 12,
        dropout: float = 0.2,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
    ):
        self.n_channels = n_channels'''

if old_pretrained_init in model_text:
    model_text = model_text.replace(old_pretrained_init, new_pretrained_init)
    print("Updated PretrainedMOMENTClassifier __init__ with n_channels=12!")

old_patch_init = '''        # 5. Patch Embedding khusus 9 Channel XAU/USD
        self.patch_embed = PatchEmbedding(
            n_channels=9,
            seq_len=64,
            patch_len=8,
            patch_stride=8,
            d_model=self.d_model,
            dropout=dropout,
        )'''

new_patch_init = '''        # 5. Patch Embedding khusus 12 Channel XAU/USD (MA Cross, SMI, SMC)
        self.patch_embed = PatchEmbedding(
            n_channels=n_channels,
            seq_len=64,
            patch_len=8,
            patch_stride=8,
            d_model=self.d_model,
            dropout=dropout,
        )'''

if old_patch_init in model_text:
    model_text = model_text.replace(old_patch_init, new_patch_init)
    print("Updated self.patch_embed to use n_channels in PretrainedMOMENTClassifier!")

p_model.write_text(model_text, encoding='utf-8')
print("All code updates completed!")
