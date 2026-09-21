"""Model C: Regularized Deep Multi-Task Sequence Model.

Architecture:
- RevIN (Reversible Instance Normalization) per sequence window.
- Compact 1D Dilated Residual CNN + LayerNorm + GRU backbone.
- Strong regularization: high dropout (0.35), weight decay (1e-3), gradient clipping, and early stopping.
- Multi-task heads:
  1. Direction classification head: P(long)
  2. Forward return regression head: Expected 12h return
  3. Realized volatility regression head: Expected 12h volatility
- Feature whitelist and anti-leakage assertions enforced.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    Dataset = None

from src.features.feature_pipeline import validate_feature_columns
from src.utils.config import ModelCConfig, resolve_device


class RevIN(nn.Module if nn is not None else object):
    """Reversible Instance Normalization to handle distribution shift in financial time series."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        if nn is not None:
            super().__init__()
            self.eps = eps
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x shape: (batch_size, seq_len, num_features)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta


class MultiTaskSequenceNetwork(nn.Module if nn is not None else object):
    """Compact, heavily regularized network with 1D Conv + LayerNorm + GRU and multi-task heads."""

    def __init__(self, in_features: int, d_model: int = 32, n_layers: int = 1, dropout: float = 0.35):
        if nn is None:
            return
        super().__init__()
        self.revin = RevIN(in_features)

        # 1D Temporal Convolution projection
        self.input_proj = nn.Conv1d(in_features, d_model, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=d_model)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        # Dilated Convolution block
        self.dilated_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=d_model)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Compact recurrent layer
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        # Multi-task heads
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self.fwd_ret_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.fwd_vol_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        # x shape: (B, L, F)
        x_norm = self.revin(x)
        # Permute to (B, F, L) for Conv1d
        x_conv = x_norm.permute(0, 2, 1)

        h = self.drop1(self.act1(self.norm1(self.input_proj(x_conv))))
        h_res = self.drop2(self.act2(self.norm2(self.dilated_conv(h))))
        h = h + h_res

        # Permute back to (B, L, d_model) for GRU
        h = h.permute(0, 2, 1)
        gru_out, _ = self.gru(h)
        # Take last time step
        last_step = gru_out[:, -1, :]

        dir_prob = self.direction_head(last_step).squeeze(-1)
        fwd_ret = self.fwd_ret_head(last_step).squeeze(-1)
        fwd_vol = self.fwd_vol_head(last_step).squeeze(-1)

        return dir_prob, fwd_ret, fwd_vol


class TimeSeriesSequenceDataset(Dataset if Dataset is not None else object):
    """Sliding window sequence dataset for multi-task time series modeling."""

    def __init__(
        self,
        features: np.ndarray,
        labels_binary: np.ndarray,
        fwd_returns: np.ndarray,
        realized_vols: np.ndarray,
        seq_len: int = 72,
    ):
        self.features = features
        self.labels_binary = labels_binary
        self.fwd_returns = fwd_returns
        self.realized_vols = realized_vols
        self.seq_len = seq_len
        self.n_samples = max(0, len(features) - seq_len + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        start = idx
        end = idx + self.seq_len
        x = self.features[start:end]
        target_idx = end - 1

        return {
            "features": torch.tensor(x, dtype=torch.float32),
            "label_binary": torch.tensor(self.labels_binary[target_idx], dtype=torch.float32),
            "fwd_ret": torch.tensor(self.fwd_returns[target_idx], dtype=torch.float32),
            "realized_vol": torch.tensor(self.realized_vols[target_idx], dtype=torch.float32),
        }


class DeepSequenceModel:
    """Wrapper managing training, checkpointing, and inference for Model C."""

    def __init__(self, config: Optional[ModelCConfig] = None):
        self.config = config or ModelCConfig()
        self.device = resolve_device("auto")
        self.model: Optional[MultiTaskSequenceNetwork] = None
        self.feature_cols: List[str] = []
        self._fitted = False

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_cols: List[str],
        save_dir: str = "models_store",
        reports_dir: str = "reports",
    ) -> "DeepSequenceModel":
        """Fit regularized sequence network with early stopping on validation loss."""
        if torch is None or nn is None:
            logger.warning("PyTorch not installed. Operating in mock mode.")
            self._fitted = True
            return self

        # Assert zero leakage in feature inputs
        validate_feature_columns(feature_cols)
        self.feature_cols = feature_cols

        # Extract numpy arrays
        X_tr = df_train[feature_cols].values
        y_tr = df_train["target_binary_long"].values
        ret_tr = df_train.get("target_fwd_ret", pd.Series(0.0, index=df_train.index)).values
        vol_tr = df_train.get("realized_vol_12", pd.Series(0.01, index=df_train.index)).values

        X_val = df_val[feature_cols].values
        y_val = df_val["target_binary_long"].values
        ret_val = df_val.get("target_fwd_ret", pd.Series(0.0, index=df_val.index)).values
        vol_val = df_val.get("realized_vol_12", pd.Series(0.01, index=df_val.index)).values

        train_ds = TimeSeriesSequenceDataset(X_tr, y_tr, ret_tr, vol_tr, seq_len=self.config.seq_len)
        val_ds = TimeSeriesSequenceDataset(X_val, y_val, ret_val, vol_val, seq_len=self.config.seq_len)

        if len(train_ds) < 100 or len(val_ds) < 50:
            logger.warning("Insufficient sequence samples for deep model training. Skipping.")
            self._fitted = True
            return self

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

        self.model = MultiTaskSequenceNetwork(
            in_features=len(feature_cols),
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        bce_loss_fn = nn.BCELoss()
        huber_loss_fn = nn.HuberLoss(delta=0.01)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)

        best_val_loss = float("inf")
        patience = self.config.early_stopping_patience
        patience_counter = 0

        train_loss_history = []
        val_loss_history = []

        save_path = Path(save_dir) / "model_c_best.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Training Model C (Deep Sequence) on {len(train_ds):,} sequences (epochs={self.config.epochs}, device={self.device})..."
        )

        for epoch in range(self.config.epochs):
            self.model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                feats = batch["features"].to(self.device)
                y_bin = batch["label_binary"].to(self.device)
                y_ret = batch["fwd_ret"].to(self.device)
                y_vol = batch["realized_vol"].to(self.device)

                optimizer.zero_grad()
                pred_dir, pred_ret, pred_vol = self.model(feats)

                l_dir = bce_loss_fn(pred_dir, y_bin)
                l_ret = huber_loss_fn(pred_ret, y_ret)
                l_vol = huber_loss_fn(pred_vol, y_vol)

                # Balanced multi-task loss
                loss = l_dir + 0.5 * l_ret + 0.2 * l_vol
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()

            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)

            # Validation
            self.model.eval()
            total_val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    feats = batch["features"].to(self.device)
                    y_bin = batch["label_binary"].to(self.device)
                    y_ret = batch["fwd_ret"].to(self.device)
                    y_vol = batch["realized_vol"].to(self.device)

                    pred_dir, pred_ret, pred_vol = self.model(feats)
                    l_dir = bce_loss_fn(pred_dir, y_bin)
                    l_ret = huber_loss_fn(pred_ret, y_ret)
                    l_vol = huber_loss_fn(pred_vol, y_vol)
                    loss = l_dir + 0.5 * l_ret + 0.2 * l_vol
                    total_val_loss += loss.item()

                    val_preds.extend(pred_dir.cpu().numpy())
                    val_targets.extend(y_bin.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)

            # Compute validation AUC
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or patience_counter >= patience:
                logger.info(
                    f"Epoch {epoch:2d}/{self.config.epochs} | Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (best val loss: {best_val_loss:.4f}).")
                break

        # Reload best model weights
        if save_path.exists():
            self.model.load_state_dict(torch.load(save_path, map_location=self.device, weights_only=True))

        self._fitted = True

        # Save loss progression curve artifact
        reports_p = Path(reports_dir)
        reports_p.mkdir(parents=True, exist_ok=True)
        with open(reports_p / "model_c_curves.json", "w") as f:
            json.dump(
                {
                    "train_loss": train_loss_history,
                    "val_loss": val_loss_history,
                    "best_val_loss": best_val_loss,
                },
                f,
                indent=2,
            )

        return self

    def load(self, checkpoint_path: Any, feature_cols: List[str]) -> "DeepSequenceModel":
        """Load pretrained model weights from checkpoint."""
        if torch is None or nn is None:
            self._fitted = True
            return self

        validate_feature_columns(feature_cols)
        path = Path(checkpoint_path)
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            if "revin.gamma" in checkpoint:
                ckpt_in_features = checkpoint["revin.gamma"].shape[0]
                if len(feature_cols) != ckpt_in_features:
                    feature_cols = feature_cols[:ckpt_in_features]

            self.feature_cols = feature_cols
            self.model = MultiTaskSequenceNetwork(
                in_features=len(feature_cols),
                d_model=self.config.d_model,
                n_layers=self.config.n_layers,
                dropout=self.config.dropout,
            ).to(self.device)

            self.model.load_state_dict(checkpoint)
            self.model.eval()
            self._fitted = True
            logger.info(f"Loaded DeepSequenceModel weights from {path} ({len(feature_cols)} features)")
        else:
            logger.warning(f"Checkpoint {path} not found. Operating in uninitialized mode.")

        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate direction probability and expected forward return."""
        if not self._fitted or self.model is None or torch is None:
            n = len(df)
            return pd.DataFrame(
                {
                    "model_c_dir_prob": np.full(n, 0.5),
                    "model_c_fwd_ret": np.zeros(n),
                    "model_c_fwd_vol": np.full(n, 0.01),
                },
                index=df.index,
            )

        self.model.eval()
        X = df[self.feature_cols].values
        seq_len = self.config.seq_len
        n = len(df)

        dir_probs = np.full(n, 0.5)
        fwd_rets = np.zeros(n)
        fwd_vols = np.full(n, 0.01)

        if n < seq_len:
            return pd.DataFrame(
                {"model_c_dir_prob": dir_probs, "model_c_fwd_ret": fwd_rets, "model_c_fwd_vol": fwd_vols},
                index=df.index,
            )

        # Batch rolling sequence inference
        windows = []
        valid_indices = []
        for i in range(seq_len - 1, n):
            windows.append(X[i - seq_len + 1 : i + 1])
            valid_indices.append(i)

        windows_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_dirs, pred_rets, pred_vols = self.model(windows_tensor)
            dir_probs[valid_indices] = pred_dirs.cpu().numpy()
            fwd_rets[valid_indices] = pred_rets.cpu().numpy()
            fwd_vols[valid_indices] = pred_vols.cpu().numpy()

        return pd.DataFrame(
            {"model_c_dir_prob": dir_probs, "model_c_fwd_ret": fwd_rets, "model_c_fwd_vol": fwd_vols},
            index=df.index,
        )
