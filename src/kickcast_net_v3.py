"""
kickcast_net_v3.py -- KickCastNet v3: Dual-path attention network with threshold tuning.

Improvements over v2:
1. Attention-weighted pooling (replaces mean pool)
2. Residual bypass path (direct MLP path that skips attention)
3. Built-in post-hoc threshold tuning
4. Fewer snapshots (5 instead of 8)
5. Feature gating (sigmoid gates learn per-dimension importance)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar, differential_evolution
from copy import deepcopy


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma=None, alpha=None):
        super().__init__()
        self.gamma = gamma if gamma is not None else torch.tensor([1.5, 2.5, 1.5])
        self.alpha = alpha if alpha is not None else torch.tensor([1.0, 2.0, 1.2])

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        pt = (probs * F.one_hot(targets, n_classes).float()).sum(dim=1)
        gamma_t = self.gamma.to(logits.device)[targets]
        alpha_t = self.alpha.to(logits.device)[targets]
        ce = F.cross_entropy(logits, targets, reduction='none')
        return (alpha_t * (1 - pt) ** gamma_t * ce).mean()


class GaussianNoise(nn.Module):
    def __init__(self, std=0.03):
        super().__init__()
        self.std = std
    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim), nn.GELU(), nn.Linear(dim, dim),
            nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.block(x)


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(2, d_token) for _ in range(n_features)])
        self.norm = nn.LayerNorm(d_token)
    def forward(self, values, masks):
        tokens = []
        for i, proj in enumerate(self.projections):
            tokens.append(proj(torch.stack([values[:, i], masks[:, i]], dim=1)))
        return self.norm(torch.stack(tokens, dim=1))


class FeatureAttention(nn.Module):
    def __init__(self, d_token, n_heads=4, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_token * 2, d_token),
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)
    def forward(self, tokens):
        attended, self.attn_weights = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + self.dropout(attended))
        tokens = self.norm2(tokens + self.dropout(self.ffn(tokens)))
        return tokens


class AttentionPool(nn.Module):
    """Learned query that attends over feature tokens -- replaces mean pool."""
    def __init__(self, d_token):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        self.key_proj = nn.Linear(d_token, d_token)
    def forward(self, tokens):
        keys = self.key_proj(tokens)
        query = self.query.expand(tokens.size(0), -1, -1)
        scores = torch.bmm(query, keys.transpose(1, 2)) / (keys.size(-1) ** 0.5)
        weights = F.softmax(scores, dim=-1)
        self.pool_weights = weights.detach()
        return torch.bmm(weights, tokens).squeeze(1)


class FeatureGate(nn.Module):
    """Sigmoid gates that learn per-dimension importance."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(x)


class KickCastNetV3(nn.Module):
    """Dual-path: attention + bypass, with attention pooling and feature gating."""
    def __init__(self, n_features=31, d_token=64, n_heads=2,
                 hidden_dim=128, n_blocks=3, dropout=0.3,
                 noise_std=0.05, attention_dropout=0.4):
        super().__init__()
        self.n_features = n_features
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.attention = FeatureAttention(d_token, n_heads, attention_dropout)
        self.attn_pool = AttentionPool(d_token)
        self.bypass = nn.Sequential(
            nn.Linear(n_features * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.merge = nn.Sequential(
            nn.Linear(d_token + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.GELU(),
        )
        self.gate = FeatureGate(hidden_dim)
        self.noise = GaussianNoise(noise_std)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3),
        )

    def forward(self, values, masks):
        tokens = self.tokenizer(values, masks)
        tokens = self.attention(tokens)
        path_a = self.attn_pool(tokens)
        path_b = self.bypass(torch.cat([values, masks], dim=1))
        merged = self.merge(torch.cat([path_a, path_b], dim=1))
        x = self.gate(merged)
        x = self.noise(x)
        x = self.blocks(x)
        return self.head(x)

    def get_attention_weights(self):
        if hasattr(self.attention, 'attn_weights'):
            return self.attention.attn_weights.detach().cpu().numpy()
        return None

    def get_pool_weights(self):
        if hasattr(self.attn_pool, 'pool_weights'):
            return self.attn_pool.pool_weights.detach().cpu().numpy()
        return None


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


class FootballDatasetV3(Dataset):
    def __init__(self, X_scaled, masks, y, feature_names, augment=True):
        self.X = torch.FloatTensor(X_scaled)
        self.masks = torch.FloatTensor(masks)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.delta_idx = [i for i, f in enumerate(feature_names) if f in DELTA_COLS]
        self.swap_pairs = []
        for a, b in SWAP_COLS:
            if a in feature_names and b in feature_names:
                self.swap_pairs.append((feature_names.index(a), feature_names.index(b)))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x, m, y = self.X[idx].clone(), self.masks[idx].clone(), self.y[idx].item()
        if self.augment and torch.rand(1).item() > 0.5:
            x[self.delta_idx] = -x[self.delta_idx]
            for a, b in self.swap_pairs:
                x[a], x[b] = x[b].item(), x[a].item()
                m[a], m[b] = m[b].item(), m[a].item()
            if y == 0: y = 2
            elif y == 2: y = 0
        return x, m, y


def mixup_batch(x, m, y_oh, alpha=0.4):
    if alpha <= 0: return x, m, y_oh
    lam = max(np.random.beta(alpha, alpha), 0.5)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], torch.maximum(m, m[idx]), lam*y_oh + (1-lam)*y_oh[idx]

def find_temperature(logits, labels):
    def nll(T):
        return log_loss(labels, F.softmax(torch.FloatTensor(logits/T), dim=1).numpy())
    return minimize_scalar(nll, bounds=(0.1, 5.0), method='bounded').x

def find_thresholds(probs, labels):
    def neg_f1(t):
        return -f1_score(labels, np.argmax(probs / np.array(t), axis=1), average='macro')
    r = differential_evolution(neg_f1, bounds=[(0.2,0.8),(0.1,0.6),(0.2,0.8)], seed=42, maxiter=200, tol=1e-6)
    return r.x


class KickCastTrainerV3:
    def __init__(self, n_features, feature_names,
                 d_token=64, n_heads=2, hidden_dim=128, n_blocks=3,
                 dropout=0.3, noise_std=0.05, attention_dropout=0.4,
                 lr=1e-3, weight_decay=1e-3,
                 focal_gamma_draw=2.0, focal_alpha_draw=2.5,
                 mixup_alpha=0.4,
                 n_cycles=5, epochs_per_cycle=30, patience=5):
        self.n_features = n_features
        self.feature_names = feature_names
        self.d_token = d_token
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.noise_std = noise_std
        self.attention_dropout = attention_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.focal_gamma_draw = focal_gamma_draw
        self.focal_alpha_draw = focal_alpha_draw
        self.mixup_alpha = mixup_alpha
        self.n_cycles = n_cycles
        self.epochs_per_cycle = epochs_per_cycle
        self.patience = patience
        self.scaler = StandardScaler()
        self.medians = None
        self.snapshots = []
        self.temperature = 1.0
        self.thresholds = np.array([1.0, 1.0, 1.0])
        self.classes_ = np.array([0, 1, 2])
        self.training_history = []

    def _preprocess(self, X, fit=False):
        X_np = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else np.array(X, dtype=np.float32)
        masks = (~np.isnan(X_np)).astype(np.float32)
        if fit:
            self.medians = np.nan_to_num(np.nanmedian(X_np, axis=0), nan=0.0)
        X_filled = X_np.copy()
        for col in range(X_filled.shape[1]):
            X_filled[np.isnan(X_filled[:, col]), col] = self.medians[col]
        X_scaled = self.scaler.fit_transform(X_filled) if fit else self.scaler.transform(X_filled)
        return X_scaled.astype(np.float32), masks

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        X_tr, m_tr = self._preprocess(X_train, fit=True)
        X_v, m_v = (None, None) if X_val is None else self._preprocess(X_val)
        X_v_t = None if X_v is None else torch.FloatTensor(X_v)
        m_v_t = None if m_v is None else torch.FloatTensor(m_v)

        ds = FootballDatasetV3(X_tr, m_tr, y_train, self.feature_names, augment=True)
        loader = DataLoader(ds, batch_size=256, shuffle=True)

        criterion = AdaptiveFocalLoss(
            gamma=torch.tensor([1.5, self.focal_gamma_draw, 1.5]),
            alpha=torch.tensor([1.0, self.focal_alpha_draw, 1.2]),
        )
        self.snapshots = []
        self.training_history = []
        best_overall = 0

        for cycle in range(self.n_cycles):
            model = KickCastNetV3(
                self.n_features, self.d_token, self.n_heads, self.hidden_dim,
                self.n_blocks, self.dropout, self.noise_std, self.attention_dropout,
            )
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs_per_cycle, eta_min=self.lr*0.01)
            best_f1, best_st, no_imp = 0, None, 0

            for ep in range(self.epochs_per_cycle):
                model.train()
                tot, nb = 0, 0
                for xb, mb, yb in loader:
                    opt.zero_grad()
                    if self.mixup_alpha > 0 and torch.rand(1).item() > 0.5:
                        yoh = F.one_hot(yb, 3).float()
                        xb, mb, ym = mixup_batch(xb, mb, yoh, self.mixup_alpha)
                        loss = -(ym * F.log_softmax(model(xb, mb), dim=1)).sum(1).mean()
                    else:
                        loss = criterion(model(xb, mb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    tot += loss.item(); nb += 1
                sched.step()

                if X_v is not None:
                    model.eval()
                    with torch.no_grad():
                        vf = f1_score(y_val, model(X_v_t, m_v_t).argmax(1).numpy(), average='macro')
                    self.training_history.append({'cycle': cycle, 'epoch': ep, 'loss': tot/nb, 'val_f1': vf})
                    if vf > best_f1:
                        best_f1, best_st, no_imp = vf, deepcopy(model.state_dict()), 0
                    else:
                        no_imp += 1
                    if no_imp >= self.patience: break

            if best_st: model.load_state_dict(best_st)
            self.snapshots.append(deepcopy(model))
            if best_f1 > best_overall: best_overall = best_f1
            if verbose:
                print(f"  Cycle {cycle+1}/{self.n_cycles}: best_val_F1={best_f1:.4f}  (overall={best_overall:.4f})")

        if X_v is not None:
            logits = self._raw_logits(X_v, m_v)
            self.temperature = find_temperature(logits, y_val)
            probs = F.softmax(torch.FloatTensor(logits / self.temperature), dim=1).numpy()
            self.thresholds = find_thresholds(probs, y_val)
            if verbose:
                print(f"  Temperature: {self.temperature:.3f}")
                print(f"  Thresholds: HW={self.thresholds[0]:.3f} D={self.thresholds[1]:.3f} AW={self.thresholds[2]:.3f}")
                tuned_f1 = f1_score(y_val, np.argmax(probs / self.thresholds, axis=1), average='macro')
                print(f"  Val F1 (with thresholds): {tuned_f1:.4f}")
        if verbose: print(f"  Ensemble: {len(self.snapshots)} snapshots")
        return self

    def _raw_logits(self, X_scaled, masks):
        Xt, Mt = torch.FloatTensor(X_scaled), torch.FloatTensor(masks)
        logits = []
        for m in self.snapshots:
            m.eval()
            with torch.no_grad(): logits.append(m(Xt, Mt).numpy())
        return np.mean(logits, axis=0)

    def predict_proba(self, X):
        Xs, ms = self._preprocess(X)
        p = F.softmax(torch.FloatTensor(self._raw_logits(Xs, ms) / self.temperature), dim=1).numpy()
        p = np.clip(p, 1e-6, 1.0)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X) / self.thresholds, axis=1)
