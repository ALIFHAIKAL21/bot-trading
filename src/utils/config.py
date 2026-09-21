"""Configuration schemas and loaders using Pydantic."""

from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4


class MarketConfig(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    equity_symbols: List[str] = Field(default_factory=lambda: ["BBCA.JK"])
    timeframe: str = "1h"
    horizon: int = 12
    history_days: int = 1095


class DataConfig(BaseModel):
    primary_source: str = "binance"
    fallback_source: str = "yfinance"
    cache_dir: str = "data/cache"
    gap_fill_policy: str = "ffill_zero_vol"
    include_funding_rate: bool = True
    include_open_interest: bool = True


class FeatureConfig(BaseModel):
    return_windows: List[int] = Field(default_factory=lambda: [1, 3, 6, 12, 24])
    volatility_windows: List[int] = Field(default_factory=lambda: [6, 12, 24, 72])
    atr_period: int = 14
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    volume_zscore_window: int = 24
    skew_kurtosis_windows: List[int] = Field(default_factory=lambda: [24, 72])
    htf_timeframes: List[str] = Field(default_factory=lambda: ["4h", "1d"])
    whitelist: List[str] = Field(
        default_factory=lambda: [
            "log_ret_1", "log_ret_3", "log_ret_6", "log_ret_12", "log_ret_24",
            "realized_vol_6", "realized_vol_12", "realized_vol_24", "realized_vol_72",
            "atr_14", "natr_14", "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "boll_pct_b", "boll_bandwidth", "vol_zscore_24",
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "skew_24", "kurt_24", "skew_72", "kurt_72",
            "htf_4h_trend", "htf_1d_trend",
            "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        ]
    )


class LabelConfig(BaseModel):
    tp_multiplier: float = 2.0
    sl_multiplier: float = 1.5
    vertical_barrier: int = 12
    min_return_filter: float = 0.001


class ModelAConfig(BaseModel):
    enabled: bool = True
    model_id: str = "amazon/chronos-bolt-tiny"
    context_length: int = 512
    prediction_length: int = 12
    batch_size: int = 64
    cache_predictions: bool = True


class ModelBConfig(BaseModel):
    enabled: bool = True
    n_estimators: int = 250
    learning_rate: float = 0.03
    max_depth: int = 5
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    optuna_trials: int = 15
    tabpfn_enabled: bool = False


class ModelCConfig(BaseModel):
    enabled: bool = True
    architecture: str = "conv_gru"
    seq_len: int = 72
    d_model: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    epochs: int = 25
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 5


class ModelDConfig(BaseModel):
    enabled: bool = True
    model_id: str = "ProsusAI/finbert"
    rss_feeds: List[str] = Field(
        default_factory=lambda: [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
        ]
    )
    use_sentiment_gate: bool = False
    veto_threshold: float = -0.4


class ModelEConfig(BaseModel):
    enabled: bool = True
    n_components: int = 3
    covariance_type: str = "full"
    n_iter: int = 100
    features: List[str] = Field(
        default_factory=lambda: ["log_ret_1", "realized_vol_12", "vol_zscore_24"]
    )


class ModelsConfig(BaseModel):
    model_a: ModelAConfig = Field(default_factory=ModelAConfig)
    model_b: ModelBConfig = Field(default_factory=ModelBConfig)
    model_c: ModelCConfig = Field(default_factory=ModelCConfig)
    model_d: ModelDConfig = Field(default_factory=ModelDConfig)
    model_e: ModelEConfig = Field(default_factory=ModelEConfig)


class EnsembleConfig(BaseModel):
    method: str = "logistic_regression"
    cv_splits: int = 5
    embargo_bars: int = 12
    l2_c: float = 1.0


class RiskConfig(BaseModel):
    target_annual_vol: float = 0.25
    kelly_fraction: float = 0.25
    max_leverage: float = 1.0
    max_position_pct: float = 0.40
    daily_loss_limit_pct: float = 0.03
    max_drawdown_limit_pct: float = 0.15
    cooldown_bars: int = 2
    entry_threshold: float = 0.54
    exit_threshold: float = 0.48
    expected_edge_hurdle_multiplier: float = 2.0
    round_trip_cost: float = 0.0030
    min_holding_bars: int = 3
    max_daily_trades: int = 6
    dust_rebalance_threshold: float = 0.05
    hard_stop_loss_pct: float = 0.03
    tp_multiplier: float = 2.0
    sl_multiplier: float = 1.5
    trade_horizon_bars: int = 12


class BacktestConfig(BaseModel):
    initial_capital: float = 10000.0
    taker_fee: float = 0.0010
    base_slippage: float = 0.0005
    vol_slippage_coeff: float = 0.05
    perp_funding_rate_8h: float = 0.0001
    monte_carlo_runs: int = 200


class ServiceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    db_path: str = "data/paper_trading.db"
    poll_interval_seconds: int = 60
    live_execution_enabled: bool = False


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)


def load_config(path: str = "config/config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    return AppConfig(**(raw_dict or {}))


def resolve_device(device_str: str = "auto") -> str:
    """Resolve compute device string safely."""
    try:
        import torch

        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device_str
    except ImportError:
        return "cpu"
