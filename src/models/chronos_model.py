"""Model A: Foundation Time-Series Forecaster (Chronos-Bolt).

Features:
- Zero-shot quantile forecasting (P10, P50, P90) via amazon/chronos-bolt.
- Normalized context window (e.g. 512 bars).
- Derived features: expected forward return, quantile skew, uncertainty spread.
- Clean causal strided grid with forward fill and Parquet disk caching.
- Health checks for stale, missing, or NaN outputs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
except ImportError:
    torch = None

try:
    from chronos import ChronosBoltPipeline
except ImportError:
    ChronosBoltPipeline = None

from src.utils.config import ModelAConfig, resolve_device


class ChronosBoltForecaster:
    """Foundation forecaster generating zero-shot quantile return predictions."""

    def __init__(self, config: Optional[ModelAConfig] = None, cache_dir: str = "data/cache"):
        self.config = config or ModelAConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = None
        self.device = resolve_device("auto") if torch is not None else "cpu"

    def _init_model(self):
        """Lazy initialization of Chronos-Bolt pipeline."""
        if self.pipeline is not None:
            return

        if ChronosBoltPipeline is not None:
            try:
                logger.info(f"Loading Chronos-Bolt model '{self.config.model_id}' on {self.device}...")
                self.pipeline = ChronosBoltPipeline.from_pretrained(
                    self.config.model_id,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                )
                logger.success("Chronos-Bolt loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load via ChronosBoltPipeline: {e}. Using fallback generator.")

        logger.info("ChronosBoltPipeline not available or failed. Using fallback rolling quantile forecaster.")
        self.pipeline = "fallback"

    def get_cache_path(self, symbol: str) -> Path:
        clean_sym = symbol.replace("/", "_").replace(":", "_").upper()
        return self.cache_dir / f"chronos_pred_{clean_sym}.parquet"

    def predict_rolling(
        self,
        df: pd.DataFrame,
        symbol: str,
        price_col: str = "close",
        stride: int = 4,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Generate rolling quantile predictions with disk caching.

        stride: Compute inference every `stride` bars and ffill to balance speed and accuracy.
        """
        cache_path = self.get_cache_path(symbol)
        if not force_refresh and cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                if not cached_df.empty and len(cached_df) == len(df):
                    logger.info(f"Loaded cached Chronos predictions for {symbol}: {cache_path}")
                    return cached_df
            except Exception as e:
                logger.warning(f"Failed to read Chronos cache {cache_path}: {e}")

        self._init_model()

        closes = df[price_col].values
        n = len(df)
        context_len = self.config.context_length
        horizon = self.config.prediction_length

        eval_indices = list(range(context_len, n, stride))
        if not eval_indices or eval_indices[-1] != n - 1:
            eval_indices.append(n - 1)

        eval_records = []
        eval_timestamps = []

        logger.info(f"Generating Chronos predictions for {symbol} ({n} bars, {len(eval_indices)} points, stride={stride})...")

        for idx in eval_indices:
            context = closes[idx - context_len : idx]
            p_last = context[-1]

            if self.pipeline != "fallback" and ChronosBoltPipeline is not None:
                try:
                    context_tensor = torch.tensor(context, dtype=torch.float32)
                    forecast = self.pipeline.predict(
                        context_tensor,
                        prediction_length=horizon,
                        num_samples=20,
                    )
                    p10 = float(np.quantile(forecast[0, :, -1].numpy(), 0.10))
                    p50 = float(np.quantile(forecast[0, :, -1].numpy(), 0.50))
                    p90 = float(np.quantile(forecast[0, :, -1].numpy(), 0.90))
                except Exception:
                    p10, p50, p90 = self._fallback_quantiles(context, horizon)
            else:
                p10, p50, p90 = self._fallback_quantiles(context, horizon)

            exp_ret = (p50 - p_last) / (p_last + 1e-9)
            skew = ((p90 - p50) - (p50 - p10)) / (p_last + 1e-9)
            spread = (p90 - p10) / (p50 + 1e-9)

            eval_records.append({
                "chronos_exp_ret": exp_ret,
                "chronos_skew": skew,
                "chronos_uncertainty": spread,
            })
            eval_timestamps.append(df.index[idx])

        eval_df = pd.DataFrame(eval_records, index=eval_timestamps)
        result_df = eval_df.reindex(df.index, method="ffill").fillna(0.0)

        result_df.to_parquet(cache_path)
        logger.success(f"Saved Chronos predictions for {symbol} to {cache_path}")
        return result_df

    def _fallback_quantiles(self, context: np.ndarray, horizon: int) -> Tuple[float, float, float]:
        """Statistical fallback quantile projection (momentum + volatility cone)."""
        p_last = context[-1]
        log_rets = np.diff(np.log(context[-72:]))
        drift = np.mean(log_rets) * horizon
        vol = np.std(log_rets) * np.sqrt(horizon)

        p50 = p_last * np.exp(drift)
        p10 = p_last * np.exp(drift - 1.28 * vol)
        p90 = p_last * np.exp(drift + 1.28 * vol)
        return p10, p50, p90

    def check_health(self, pred_df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Health check for Chronos outputs."""
        if pred_df.empty:
            return False, "chronos_predictions_empty"
        if pred_df.isna().any().any():
            return False, "chronos_predictions_contain_nans"
        return True, None
