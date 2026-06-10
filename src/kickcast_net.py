"""
kickcast_net.py -- KickCastNet: Custom neural network for football match prediction.

Architecture based on findings from "When Do Neural Nets Outperform Boosted Trees
on Tabular Data?" (2023) -- regularized MLP with residual blocks beats CatBoost when
you combine: BatchNorm, dropout, weight decay, skip connections, focal loss,
snapshot ensembling, and cosine annealing LR.

Key design decisions:
1. Focal loss (gamma=2) to focus on hard draws instead of easy home wins
2. Home/away augmentation built into training (online, not pre-computed)
3. Feature-wise dropout (randomly zero entire features, not just neurons)
4. Residual blocks with pre-activation BatchNorm (identity shortcuts)
5. Snapshot ensemble: save model at each LR cycle minimum, average at inference
6. Cosine annealing with warm restarts for learning rate schedule
7. Strict early stopping on VALIDATION macro F1 (not training loss)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from copy import deepcopy


# Focal Loss

class FocalLoss(nn.Module):
    """Multiclass focal loss. Down-weights easy examples, focuses on hard ones.

    For draws: the model finds draws hard, so focal loss keeps their gradient strong.
    For obvious home wins: the model is confident, so focal loss reduces their gradient.
    Net effect: more learning capacity spent on draws.
    """
    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        loss = focal_weight * ce
        return loss.mean()


# Feature-Wise Dropout

class FeatureDropout(nn.Module):
    """Drop entire features (columns), not individual neurons.
    Forces the model to not rely exclusively on elo_diff."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            mask = torch.bernoulli(torch.full((1, x.size(1)), 1 - self.p, device=x.device))
            return x * mask / (1 - self.p)
        return x


# Residual Block

class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN, ReLU, Linear, BN, ReLU, Dropout, Linear.
    Skip connection adds input directly to output (identity shortcut)."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


# KickCastNet

class KickCastNet(nn.Module):
    """Custom football prediction network.
    Architecture: FeatureDropout, Linear, [ResidualBlock x N], Linear, 3 classes
    """
    def __init__(self, n_features, hidden_dim=128, n_blocks=3,
                 dropout=0.3, feature_dropout=0.1):
        super().__init__()

        self.feature_drop = FeatureDropout(feature_dropout)
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        x = self.feature_drop(x)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)


# Dataset with Online Augmentation

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


class FootballDataset(Dataset):
    """Dataset with online home/away augmentation.
    Each epoch, every sample has a 50% chance of being flipped."""

    def __init__(self, X, y, feature_names, augment=True):
        self.X = torch.FloatTensor(X)
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
        y = self.y[idx].item()

        if self.augment and torch.rand(1).item() > 0.5:
            x[self.delta_idx] = -x[self.delta_idx]
            for a, b in self.swap_pairs:
                x[a], x[b] = x[b].item(), x[a].item()
            if y == 0:
                y = 2
            elif y == 2:
                y = 0

        return x, y


# Training Engine with Snapshot Ensembling

class KickCastTrainer:
    """Full training pipeline with snapshot ensembling and early stopping."""

    def __init__(self, n_features, feature_names,
                 hidden_dim=128, n_blocks=3, dropout=0.3,
                 feature_dropout=0.1, lr=1e-3, weight_decay=1e-4,
                 focal_gamma=2.0, class_weights=None,
                 n_cycles=5, epochs_per_cycle=40, patience=3):
        self.n_features = n_features
        self.feature_names = feature_names
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.n_cycles = n_cycles
        self.epochs_per_cycle = epochs_per_cycle
        self.patience = patience

        self.scaler = StandardScaler()
        self.snapshots = []
        self.classes_ = np.array([0, 1, 2])

        if class_weights is not None:
            self.class_weights_tensor = torch.FloatTensor(class_weights)
        else:
            self.class_weights_tensor = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train with cosine annealing + snapshot ensembling."""
        X_train_np = (pd.DataFrame(X_train).fillna(0).values
                      if isinstance(X_train, pd.DataFrame) else np.nan_to_num(X_train, 0))
        X_train_scaled = self.scaler.fit_transform(X_train_np)

        if X_val is not None:
            X_val_np = (pd.DataFrame(X_val).fillna(0).values
                        if isinstance(X_val, pd.DataFrame) else np.nan_to_num(X_val, 0))
            X_val_scaled = self.scaler.transform(X_val_np)

        train_ds = FootballDataset(X_train_scaled, y_train, self.feature_names, augment=True)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)

        self.snapshots = []
        best_overall_f1 = 0

        for cycle in range(self.n_cycles):
            model = KickCastNet(
                self.n_features, self.hidden_dim, self.n_blocks,
                self.dropout, self.feature_dropout
            )
            criterion = FocalLoss(self.focal_gamma, self.class_weights_tensor)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs_per_cycle, eta_min=self.lr * 0.01
            )

            best_val_f1 = 0
            best_state = None
            no_improve = 0

            for epoch in range(self.epochs_per_cycle):
                model.train()
                total_loss = 0
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                scheduler.step()

                if X_val is not None:
                    model.eval()
                    with torch.no_grad():
                        val_logits = model(torch.FloatTensor(X_val_scaled))
                        val_pred = val_logits.argmax(dim=1).numpy()
                        val_f1 = f1_score(y_val, val_pred, average='macro')

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

            print(f"  Cycle {cycle+1}/{self.n_cycles}: "
                  f"best_val_F1={best_val_f1:.4f}  "
                  f"(overall_best={best_overall_f1:.4f})")

        print(f"  Ensemble: {len(self.snapshots)} snapshots")
        return self

    def predict_proba(self, X):
        """Average predictions across all snapshots."""
        X_np = (pd.DataFrame(X).fillna(0).values
                if isinstance(X, pd.DataFrame) else np.nan_to_num(X, 0))
        X_scaled = self.scaler.transform(X_np)
        X_tensor = torch.FloatTensor(X_scaled)

        all_probs = []
        for model in self.snapshots:
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                probs = F.softmax(logits, dim=1).numpy()
                all_probs.append(probs)

        avg_proba = np.mean(all_probs, axis=0)
        avg_proba = np.clip(avg_proba, 1e-6, 1.0)
        avg_proba = avg_proba / avg_proba.sum(axis=1, keepdims=True)
        return avg_proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
