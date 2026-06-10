"""
kickcast_net_v2.py -- KickCastNet v2: Attention-augmented neural network for football
match prediction.

Improvements over v1:
1. Learned missing-value handling (v1 filled NaN with 0, destroying signal for 45% of
   squad value features. v2 creates explicit mask + learns fill values)
2. Feature tokenization with self-attention (captures feature interactions that GBDTs
   get for free via tree splits)
3. Adaptive per-class focal loss (draw gamma=2.5 vs home/away gamma=1.5)
4. Input-space mixup regularization (proven for small tabular data)
5. Gaussian feature noise (complementary to dropout)
6. Label smoothing for better calibration
7. Temperature scaling post-training (directly optimizes log loss)

Architecture:
    Input (31 features + 31 missing masks)
    -> Feature Tokenizer (31 tokens x d_token each)
    -> Multi-Head Self-Attention (1 layer, captures interactions)
    -> Mean Pool -> d_token vector
    -> Residual MLP blocks (same proven design from v1)
    -> Linear -> 3 class logits
    -> Temperature scaling at inference
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
from copy import deepcopy


# ── Adaptive Per-Class Focal Loss ────────────────────────────────────

class AdaptiveFocalLoss(nn.Module):
    """Focal loss with different gamma/alpha per class.

    Draws are hard: high gamma (2.5) keeps gradient strong on misclassified draws.
    Home wins are easy: low gamma (1.5) reduces their gradient dominance.
    Alpha upweights draw contribution to total loss.
    """
    def __init__(self, gamma=None, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma if gamma is not None else torch.tensor([1.5, 2.5, 1.5])
        self.alpha = alpha if alpha is not None else torch.tensor([1.0, 2.0, 1.2])
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        if self.label_smoothing > 0:
            targets_oh = F.one_hot(targets, n_classes).float()
            targets_smooth = (
                targets_oh * (1 - self.label_smoothing)
                + self.label_smoothing / n_classes
            )
        else:
            targets_smooth = F.one_hot(targets, n_classes).float()

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        pt = (probs * F.one_hot(targets, n_classes).float()).sum(dim=1)

        gamma_t = self.gamma.to(logits.device)[targets]
        alpha_t = self.alpha.to(logits.device)[targets]
        focal_weight = alpha_t * (1 - pt) ** gamma_t

        ce = -(targets_smooth * log_probs).sum(dim=1)
        return (focal_weight * ce).mean()


# ── Feature Noise ────────────────────────────────────────────────────

class GaussianNoise(nn.Module):
    """Additive Gaussian noise on inputs during training."""
    def __init__(self, std=0.03):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


# ── Feature-Wise Dropout (from v1) ──────────────────────────────────

class FeatureDropout(nn.Module):
    """Drop entire features, not individual neurons."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            mask = torch.bernoulli(
                torch.full((1, x.size(1)), 1 - self.p, device=x.device)
            )
            return x * mask / (1 - self.p)
        return x


# ── Residual Block (GELU upgrade from v1's ReLU) ────────────────────

class ResidualBlock(nn.Module):
    """Pre-activation residual: BN -> GELU -> Linear -> BN -> GELU -> Drop -> Linear."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


# ── Feature Tokenizer ───────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """Project each (value, mask) pair into a d_token embedding.
    Result: (batch, n_features, d_token) -- one token per feature."""
    def __init__(self, n_features, d_token):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(2, d_token) for _ in range(n_features)
        ])
        self.norm = nn.LayerNorm(d_token)

    def forward(self, values, masks):
        tokens = []
        for i, proj in enumerate(self.projections):
            feat_input = torch.stack([values[:, i], masks[:, i]], dim=1)
            tokens.append(proj(feat_input))
        tokens = torch.stack(tokens, dim=1)
        return self.norm(tokens)


# ── Multi-Head Self-Attention ────────────────────────────────────────

class FeatureAttention(nn.Module):
    """Single self-attention layer over feature tokens.
    Only 1 layer to avoid overfitting on 15K samples."""
    def __init__(self, d_token, n_heads=4, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * 2, d_token),
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        attended, self.attn_weights = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + self.dropout(attended))
        tokens = self.norm2(tokens + self.dropout(self.ffn(tokens)))
        return tokens


# ── KickCastNet v2 ───────────────────────────────────────────────────

class KickCastNetV2(nn.Module):
    """Full v2 architecture with attention + missing value handling."""
    def __init__(self, n_features=31, d_token=64, n_heads=4,
                 hidden_dim=128, n_blocks=3, dropout=0.3,
                 feature_dropout=0.1, noise_std=0.03,
                 attention_dropout=0.3):
        super().__init__()
        self.n_features = n_features
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.attention = FeatureAttention(d_token, n_heads, attention_dropout)
        self.pool_proj = nn.Sequential(
            nn.Linear(d_token, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.noise = GaussianNoise(noise_std)
        self.feat_drop = FeatureDropout(feature_dropout)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, values, masks):
        tokens = self.tokenizer(values, masks)
        tokens = self.attention(tokens)
        pooled = tokens.mean(dim=1)
        x = self.pool_proj(pooled)
        x = self.noise(x)
        x = self.feat_drop(x)
        x = self.blocks(x)
        return self.head(x)

    def get_attention_weights(self):
        if hasattr(self.attention, 'attn_weights') and self.attention.attn_weights is not None:
            return self.attention.attn_weights.detach().cpu().numpy()
        return None


# ── Home/Away Augmentation Columns ───────────────────────────────────

DELTA_COLS = [
    "elo_diff", "elo_momentum_diff", "rank_diff", "points_diff",
    "squad_value_total_delta", "squad_value_top11_delta",
    "squad_value_attack_delta", "squad_value_mid_delta",
    "squad_value_def_delta", "star_player_value_delta", "squad_depth_delta",
    "form_win_rate_diff", "form_weighted_diff", "goal_diff_delta",
    "h2h_home_win_rate", "tournament_wr_delta",
    "wc_appearances_diff", "wc_knockout_rate_diff",
    "wc_best_finish_diff", "wc_goals_per_game_diff",
    "injury_count_delta", "injury_burden_delta", "star_injury_flag",
]
SWAP_COLS = [("home_days_rest", "away_days_rest")]


# ── Dataset ──────────────────────────────────────────────────────────

class FootballDatasetV2(Dataset):
    """Provides (values, masks, labels) with home/away flip augmentation."""
    def __init__(self, X_scaled, masks, y, feature_names, augment=True):
        self.X = torch.FloatTensor(X_scaled)
        self.masks = torch.FloatTensor(masks)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.feature_names = feature_names
        self.delta_idx = [i for i, f in enumerate(feature_names) if f in DELTA_COLS]
        self.swap_pairs = []
        for a, b in SWAP_COLS:
            if a in feature_names and b in feature_names:
                self.swap_pairs.append((feature_names.index(a), feature_names.index(b)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        m = self.masks[idx].clone()
        y = self.y[idx].item()
        if self.augment and torch.rand(1).item() > 0.5:
            x[self.delta_idx] = -x[self.delta_idx]
            for a, b in self.swap_pairs:
                x[a], x[b] = x[b].item(), x[a].item()
                m[a], m[b] = m[b].item(), m[a].item()
            if y == 0:
                y = 2
            elif y == 2:
                y = 0
        return x, m, y


# ── Mixup ────────────────────────────────────────────────────────────

def mixup_batch(x, m, y_onehot, alpha=0.2):
    """Input-space mixup: interpolate between pairs of samples."""
    if alpha <= 0:
        return x, m, y_onehot
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_m = torch.maximum(m, m[idx])
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return mixed_x, mixed_m, mixed_y


# ── Temperature Scaling ──────────────────────────────────────────────

def find_temperature(logits, labels):
    """Find optimal temperature T that minimizes NLL on validation set."""
    def nll(T):
        scaled = logits / T
        probs = F.softmax(torch.FloatTensor(scaled), dim=1).numpy()
        return log_loss(labels, probs)
    result = minimize_scalar(nll, bounds=(0.1, 5.0), method='bounded')
    return result.x


# ── Training Engine ──────────────────────────────────────────────────

class KickCastTrainerV2:
    """Full training pipeline for KickCastNet v2 with snapshot ensembling."""

    def __init__(self, n_features, feature_names,
                 d_token=64, n_heads=4,
                 hidden_dim=128, n_blocks=3, dropout=0.3,
                 feature_dropout=0.1, noise_std=0.03,
                 attention_dropout=0.3,
                 lr=1e-3, weight_decay=1e-4,
                 focal_gamma_draw=2.5, focal_alpha_draw=2.0,
                 label_smoothing=0.05, mixup_alpha=0.2,
                 class_weights=None,
                 n_cycles=5, epochs_per_cycle=40, patience=5):
        self.n_features = n_features
        self.feature_names = feature_names
        self.d_token = d_token
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.noise_std = noise_std
        self.attention_dropout = attention_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.focal_gamma_draw = focal_gamma_draw
        self.focal_alpha_draw = focal_alpha_draw
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.n_cycles = n_cycles
        self.epochs_per_cycle = epochs_per_cycle
        self.patience = patience
        self.scaler = StandardScaler()
        self.medians = None
        self.snapshots = []
        self.temperature = 1.0
        self.classes_ = np.array([0, 1, 2])
        self.training_history = []

    def _preprocess(self, X, fit=False):
        """Convert to (scaled_values, masks). Handles NaN properly."""
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.array(X, dtype=np.float32)

        masks = (~np.isnan(X_np)).astype(np.float32)

        if fit:
            self.medians = np.nanmedian(X_np, axis=0)
            self.medians = np.nan_to_num(self.medians, nan=0.0)

        X_filled = X_np.copy()
        for col in range(X_filled.shape[1]):
            nan_idx = np.isnan(X_filled[:, col])
            X_filled[nan_idx, col] = self.medians[col]

        if fit:
            X_scaled = self.scaler.fit_transform(X_filled)
        else:
            X_scaled = self.scaler.transform(X_filled)

        return X_scaled.astype(np.float32), masks

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        X_train_scaled, train_masks = self._preprocess(X_train, fit=True)

        if X_val is not None:
            X_val_scaled, val_masks = self._preprocess(X_val, fit=False)
            X_val_t = torch.FloatTensor(X_val_scaled)
            val_masks_t = torch.FloatTensor(val_masks)

        train_ds = FootballDatasetV2(
            X_train_scaled, train_masks, y_train, self.feature_names, augment=True,
        )
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)

        focal_gamma = torch.tensor([1.5, self.focal_gamma_draw, 1.5])
        focal_alpha = torch.tensor([1.0, self.focal_alpha_draw, 1.2])
        criterion = AdaptiveFocalLoss(
            gamma=focal_gamma, alpha=focal_alpha,
            label_smoothing=self.label_smoothing,
        )

        self.snapshots = []
        self.training_history = []
        best_overall_f1 = 0

        for cycle in range(self.n_cycles):
            model = KickCastNetV2(
                n_features=self.n_features, d_token=self.d_token,
                n_heads=self.n_heads, hidden_dim=self.hidden_dim,
                n_blocks=self.n_blocks, dropout=self.dropout,
                feature_dropout=self.feature_dropout, noise_std=self.noise_std,
                attention_dropout=self.attention_dropout,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs_per_cycle, eta_min=self.lr * 0.01,
            )

            best_val_f1 = 0
            best_state = None
            no_improve = 0

            for epoch in range(self.epochs_per_cycle):
                model.train()
                total_loss = 0
                n_batches = 0

                for xb, mb, yb in train_loader:
                    optimizer.zero_grad()

                    if self.mixup_alpha > 0 and torch.rand(1).item() > 0.5:
                        yb_onehot = F.one_hot(yb, 3).float()
                        xb, mb, yb_mixed = mixup_batch(xb, mb, yb_onehot, self.mixup_alpha)
                        logits = model(xb, mb)
                        log_probs = F.log_softmax(logits, dim=1)
                        loss = -(yb_mixed * log_probs).sum(dim=1).mean()
                    else:
                        logits = model(xb, mb)
                        loss = criterion(logits, yb)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                val_f1 = 0
                if X_val is not None:
                    model.eval()
                    with torch.no_grad():
                        val_logits = model(X_val_t, val_masks_t)
                        val_pred = val_logits.argmax(dim=1).numpy()
                        val_f1 = f1_score(y_val, val_pred, average='macro')

                    self.training_history.append({
                        'cycle': cycle, 'epoch': epoch,
                        'loss': total_loss / n_batches, 'val_f1': val_f1,
                        'lr': scheduler.get_last_lr()[0],
                    })

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_state = deepcopy(model.state_dict())
                        no_improve = 0
                    else:
                        no_improve += 1

                    if no_improve >= self.patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            self.snapshots.append(deepcopy(model))

            if best_val_f1 > best_overall_f1:
                best_overall_f1 = best_val_f1

            if verbose:
                print(f"  Cycle {cycle+1}/{self.n_cycles}: "
                      f"best_val_F1={best_val_f1:.4f}  "
                      f"(overall_best={best_overall_f1:.4f})")

        if X_val is not None:
            all_logits = self._raw_logits(X_val_scaled, val_masks)
            self.temperature = find_temperature(all_logits, y_val)
            if verbose:
                print(f"  Temperature: {self.temperature:.3f}")

        if verbose:
            print(f"  Ensemble: {len(self.snapshots)} snapshots")

        return self

    def _raw_logits(self, X_scaled, masks):
        X_t = torch.FloatTensor(X_scaled)
        M_t = torch.FloatTensor(masks)
        all_logits = []
        for model in self.snapshots:
            model.eval()
            with torch.no_grad():
                logits = model(X_t, M_t).numpy()
                all_logits.append(logits)
        return np.mean(all_logits, axis=0)

    def predict_proba(self, X):
        X_scaled, masks = self._preprocess(X, fit=False)
        avg_logits = self._raw_logits(X_scaled, masks)
        scaled_logits = avg_logits / self.temperature
        probs = F.softmax(torch.FloatTensor(scaled_logits), dim=1).numpy()
        probs = np.clip(probs, 1e-6, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def get_attention_weights(self, X):
        X_scaled, masks = self._preprocess(X, fit=False)
        X_t = torch.FloatTensor(X_scaled)
        M_t = torch.FloatTensor(masks)
        model = self.snapshots[0]
        model.eval()
        with torch.no_grad():
            _ = model(X_t, M_t)
            return model.get_attention_weights()
