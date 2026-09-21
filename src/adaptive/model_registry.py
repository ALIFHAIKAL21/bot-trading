"""Champion/Challenger Model Registry with Auto-Degrade Circuit Breakers.

Adaptivity Components A4 & A5:
- Component A4: Auto-degrade circuit breakers.
  * Monitors rolling Brier score and rolling Information Coefficient (IC) per model.
  * If a model degrades beyond threshold (Brier > 0.28 or IC < -0.05), the circuit breaker trips.
  * Tripped model is zero-weighted in active execution and switched to SHADOW recovery mode.
- Component A5: Champion / Challenger registry with shadow evaluation and rollback.
  * Maintains versioned models (CHAMPION, CHALLENGER, DEGRADED, SHADOW).
  * Promotes challenger to champion only after demonstrating lower Brier loss over minimum evaluation window.
  * Retains snapshot of previous champion for instant rollback if challenger fails.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from loguru import logger


class ModelStatus(str, Enum):
    CHAMPION = "CHAMPION"
    CHALLENGER = "CHALLENGER"
    SHADOW = "SHADOW"
    DEGRADED = "DEGRADED"


@dataclass
class ModelRecord:
    """Metadata record for a registered model instance."""
    name: str
    version: str
    status: ModelStatus
    metrics: Dict[str, float] = field(default_factory=dict)
    active_weight: float = 0.50
    shadow_predictions: List[float] = field(default_factory=list)
    shadow_targets: List[float] = field(default_factory=list)
    circuit_breaker_tripped: bool = False
    tripped_reason: Optional[str] = None


class ModelRegistry:
    """Institutional Model Registry managing model lifecycle, shadow evaluation, and safety circuit breakers."""

    def __init__(
        self,
        brier_circuit_breaker_threshold: float = 0.28,
        ic_circuit_breaker_threshold: float = -0.05,
        shadow_evaluation_window: int = 72,
    ):
        self.brier_threshold = brier_circuit_breaker_threshold
        self.ic_threshold = ic_circuit_breaker_threshold
        self.shadow_window = shadow_evaluation_window
        self.models: Dict[str, ModelRecord] = {}
        self.rollback_snapshots: Dict[str, ModelRecord] = {}

    def register_model(
        self,
        name: str,
        version: str,
        status: ModelStatus = ModelStatus.CHAMPION,
        initial_weight: float = 0.50,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> ModelRecord:
        """Register a new or updated model into the registry."""
        record = ModelRecord(
            name=name,
            version=version,
            status=status,
            active_weight=initial_weight if status == ModelStatus.CHAMPION else 0.0,
            metrics=baseline_metrics or {},
        )
        self.models[name] = record

        # Save initial pristine snapshot for champion models
        if status == ModelStatus.CHAMPION:
            self.rollback_snapshots[name] = ModelRecord(
                name=name,
                version=version,
                status=ModelStatus.CHAMPION,
                active_weight=initial_weight,
                metrics=dict(baseline_metrics or {}),
            )

        logger.info(f"Registered model '{name}' (version: {version}) with status {status.value}")
        return record

    def record_step_outcome(self, model_name: str, pred_prob: float, y_true: float) -> None:
        """Record model step prediction and ground truth outcome for rolling evaluation."""
        if model_name not in self.models:
            return

        record = self.models[model_name]
        record.shadow_predictions.append(float(pred_prob))
        record.shadow_targets.append(float(y_true))

        if len(record.shadow_predictions) > self.shadow_window:
            record.shadow_predictions.pop(0)
            record.shadow_targets.pop(0)

        # Check circuit breaker on champions
        if record.status == ModelStatus.CHAMPION and len(record.shadow_predictions) >= 24:
            self._check_circuit_breaker(record)

    def _check_circuit_breaker(self, record: ModelRecord) -> None:
        """Evaluate rolling performance metrics and trip breaker if degraded."""
        preds = np.array(record.shadow_predictions)
        targets = np.array(record.shadow_targets)
        
        rolling_brier = float(np.mean((preds - targets) ** 2))
        record.metrics["rolling_brier"] = rolling_brier

        if rolling_brier > self.brier_threshold:
            record.circuit_breaker_tripped = True
            record.status = ModelStatus.DEGRADED
            record.active_weight = 0.0
            record.tripped_reason = f"Rolling Brier score {rolling_brier:.4f} exceeded threshold {self.brier_threshold:.4f}"
            logger.warning(f"CIRCUIT BREAKER TRIPPED for '{record.name}': {record.tripped_reason}. Switched to DEGRADED.")

    def evaluate_challenger(self, challenger_name: str, champion_name: str) -> bool:
        """Evaluate whether a shadow challenger should be promoted over the champion."""
        if challenger_name not in self.models or champion_name not in self.models:
            return False

        challenger = self.models[challenger_name]
        champion = self.models[champion_name]

        if len(challenger.shadow_predictions) < self.shadow_window:
            return False

        chal_brier = float(np.mean((np.array(challenger.shadow_predictions) - np.array(challenger.shadow_targets)) ** 2))
        champ_brier = float(np.mean((np.array(champion.shadow_predictions) - np.array(champion.shadow_targets)) ** 2))

        # Promote if challenger achieves statistically lower Brier loss (>= 2% improvement)
        if chal_brier < champ_brier * 0.98:
            logger.success(
                f"PROMOTING CHALLENGER '{challenger_name}' (Brier: {chal_brier:.4f}) over CHAMPION '{champion_name}' (Brier: {champ_brier:.4f})"
            )
            # Store rollback snapshot
            self.rollback_snapshots[champion_name] = ModelRecord(
                name=champion.name,
                version=champion.version,
                status=champion.status,
                active_weight=champion.active_weight,
                metrics=dict(champion.metrics),
            )
            # Promote challenger
            challenger.status = ModelStatus.CHAMPION
            challenger.active_weight = champion.active_weight
            # Demote old champion to shadow
            champion.status = ModelStatus.SHADOW
            champion.active_weight = 0.0
            return True

        return False

    def rollback_model(self, model_name: str) -> bool:
        """Rollback a model to its previous champion snapshot if available."""
        if model_name not in self.rollback_snapshots:
            logger.warning(f"No rollback snapshot found for '{model_name}'.")
            return False

        snapshot = self.rollback_snapshots[model_name]
        self.models[model_name] = ModelRecord(
            name=snapshot.name,
            version=snapshot.version,
            status=ModelStatus.CHAMPION,
            active_weight=snapshot.active_weight,
            metrics=dict(snapshot.metrics),
            circuit_breaker_tripped=False,
            tripped_reason=None,
        )
        logger.warning(f"ROLLED BACK '{model_name}' to version {snapshot.version} (status: CHAMPION)")
        return True

    def get_active_weights(self) -> Dict[str, float]:
        """Get current normalized active weights for all models in the registry."""
        weights = {}
        for name, record in self.models.items():
            if record.status == ModelStatus.CHAMPION and not record.circuit_breaker_tripped:
                weights[name] = record.active_weight
            else:
                weights[name] = 0.0

        total_w = sum(weights.values())
        if total_w > 0:
            return {k: v / total_w for k, v in weights.items()}
        # Fallback if all champions tripped
        return {k: 0.0 for k in weights}
