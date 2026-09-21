r"""Online Causal Model Weighting via Hedge / Exponentially Weighted Average (EWA).

Adaptivity Component A2:
- Causal online Hedge / EWA algorithm for dynamic model re-weighting.
- Maintains discounted cumulative loss per model using Brier score: L_m(t) = lambda * L_m(t-1) + (p_m - y)^2.
- Updates weights causally: w_m(t) \propto exp(-eta * L_m(t)).
- Strictly causal: update only occurs when ground truth label y is fully resolved (post-embargo).
- Guarantees convex combination: sum(w_m) = 1.0, w_m >= min_weight.
"""

from typing import Dict, List, Optional
import numpy as np
from loguru import logger


class OnlineWeightManager:
    """Dynamic online ensemble weight manager implementing the Hedge/EWA algorithm."""

    def __init__(
        self,
        models: List[str],
        eta: float = 2.0,
        decay_factor: float = 0.995,
        min_weight: float = 0.05,
    ):
        if not models:
            raise ValueError("Models list cannot be empty.")
        self.models = list(models)
        self.eta = eta
        self.decay_factor = decay_factor
        self.min_weight = min_weight
        self.reset()

    def reset(self) -> None:
        """Reset weights to equal distribution and clear cumulative losses."""
        n = len(self.models)
        self.weights: Dict[str, float] = {m: 1.0 / n for m in self.models}
        self.cum_losses: Dict[str, float] = {m: 0.0 for m in self.models}
        self.loss_history: List[Dict[str, float]] = []
        self.weight_history: List[Dict[str, float]] = [{m: 1.0 / n for m in self.models}]

    def update(self, y_true: float, predictions: Dict[str, float]) -> Dict[str, float]:
        """Causally update model weights upon observing true resolved outcome.
        
        y_true: 1.0 for positive barrier hit, 0.0 for negative.
        predictions: Dict mapping model_name -> predicted probability in [0, 1].
        """
        # 1. Compute Brier loss for each model: (p - y)^2
        step_losses = {}
        for m in self.models:
            p = float(predictions.get(m, 0.5))
            loss = (p - y_true) ** 2
            step_losses[m] = loss
            # Update discounted cumulative loss
            self.cum_losses[m] = self.decay_factor * self.cum_losses[m] + loss

        self.loss_history.append(step_losses)

        # 2. Re-compute weights using softmax over negative scaled cumulative losses
        losses_arr = np.array([self.cum_losses[m] for m in self.models])
        min_loss = np.min(losses_arr)
        scaled = -self.eta * (losses_arr - min_loss)
        exp_weights = np.exp(scaled)
        norm_weights = exp_weights / np.sum(exp_weights)

        # Apply strict minimum weight floor
        if self.min_weight > 0 and len(self.models) * self.min_weight < 1.0:
            norm_weights = np.maximum(norm_weights, self.min_weight)
            # Re-normalize excess above floor
            excess = norm_weights - self.min_weight
            norm_weights = self.min_weight + (excess / np.sum(excess)) * (1.0 - len(self.models) * self.min_weight)

        for i, m in enumerate(self.models):
            self.weights[m] = float(norm_weights[i])

        self.weight_history.append(dict(self.weights))
        return dict(self.weights)

    def get_weights(self) -> Dict[str, float]:
        """Retrieve current model weights."""
        return dict(self.weights)

    def combine_predictions(self, predictions: Dict[str, float]) -> float:
        """Compute convex combination of current predictions using online weights."""
        combined_prob = 0.0
        for m in self.models:
            w = self.weights.get(m, 0.0)
            p = float(predictions.get(m, 0.5))
            combined_prob += w * p
        return float(np.clip(combined_prob, 0.0, 1.0))
