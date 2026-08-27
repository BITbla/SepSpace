"""
Model definitions for SepSpace experiments.
Includes: Channel Encoder, SepSpace model, baseline heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelEncoder(nn.Module):
    """Transformer-based Channel Encoder from prior work."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, seq_len=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(seq_len + 1, d_model)  # +1 for CLS
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: (B, L) token IDs
        Returns:
            h: (B, d_model) CLS token embedding
        """
        B, L = x.shape
        cls_tok = torch.ones(B, 1, dtype=torch.long, device=x.device)
        x = torch.cat([cls_tok, x], dim=1)  # (B, L+1)
        pos_ids = torch.arange(L + 1, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos_ids)
        h = self.transformer(h)
        return h[:, 0]  # CLS token


class SepSpaceModel(nn.Module):
    """
    SepSpace: CE+SupCon with prototype-radius rejection.

    Components:
    - encoder: Channel Encoder
    - classifier: Linear classifier (for CE loss)
    - proj_head: 2-layer MLP projection head (for SupCon loss, training only)
    """

    def __init__(self, vocab_size, n_classes, d_model=64, d_proj=32):
        super().__init__()
        self.encoder = ChannelEncoder(vocab_size, d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_proj)
        )

    def forward(self, x):
        """Returns encoder embedding (for inference)."""
        return self.encoder(x)

    def classify(self, x):
        """Returns (logits, embedding)."""
        h = self.encoder(x)
        return self.classifier(h), h

    def project(self, x):
        """Returns L2-normalized projection (for SupCon)."""
        h = self.encoder(x)
        z = self.proj_head(h)
        return F.normalize(z, dim=-1)


class BaselineModel(nn.Module):
    """
    Baseline model: CE-only training.
    Used for MSP, Energy, Mahalanobis baselines.
    """

    def __init__(self, vocab_size, n_classes, d_model=64):
        super().__init__()
        self.encoder = ChannelEncoder(vocab_size, d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """Returns encoder embedding."""
        return self.encoder(x)

    def classify(self, x):
        """Returns (logits, embedding)."""
        h = self.encoder(x)
        return self.classifier(h), h


def supcon_loss(z, labels, temp=0.07):
    """
    Supervised Contrastive Loss (Khosla et al. NeurIPS 2020).

    Args:
        z: (B, d) L2-normalized embeddings
        labels: (B,) class labels
        temp: temperature

    Returns:
        loss: scalar
    """
    B = z.shape[0]
    sim = torch.mm(z, z.T) / temp  # (B, B)

    # Mask out self
    mask_self = torch.eye(B, dtype=torch.bool, device=z.device)

    # Positive mask: same label, not self
    labels = labels.view(-1, 1)
    mask_pos = (labels == labels.T) & ~mask_self  # (B, B)

    # Numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim)
    exp_sim_no_self = exp_sim * (~mask_self).float()
    log_prob = sim - torch.log(exp_sim_no_self.sum(dim=1, keepdim=True) + 1e-8)

    # Mean over positives
    n_pos = mask_pos.float().sum(dim=1)
    loss = -(log_prob * mask_pos.float()).sum(dim=1)
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    loss = (loss[valid] / n_pos[valid]).mean()
    return loss


def train_stage1(model, loader, epochs, lr, device):
    """
    Stage 1: CE-only training.

    Args:
        model: SepSpaceModel or BaselineModel
        loader: DataLoader
        epochs: number of epochs
        lr: learning rate
        device: torch device
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for ep in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits, _ = model.classify(X)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        if (ep + 1) % 5 == 0:
            acc = correct / total
            print(f"  Stage1 ep{ep+1:02d}: loss={total_loss/len(loader):.4f} acc={acc:.4f}")


def train_stage2(model, loader, epochs, lr, lambda_sup, device):
    """
    Stage 2: CE + SupCon joint fine-tuning.
    Classifier frozen, encoder + proj_head trainable.

    Args:
        model: SepSpaceModel
        loader: DataLoader
        epochs: number of epochs
        lr: learning rate (typically 0.3 * stage1_lr)
        lambda_sup: weight for SupCon loss
        device: torch device
    """
    # Freeze classifier
    for p in model.classifier.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.proj_head.parameters()),
        lr=lr
    )
    model.train()

    for ep in range(epochs):
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits, _ = model.classify(X)
            loss_ce = F.cross_entropy(logits, y)
            z = model.project(X)
            loss_sc = supcon_loss(z, y)
            loss = loss_ce + lambda_sup * loss_sc

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        if (ep + 1) % 5 == 0:
            print(f"  Stage2 ep{ep+1:02d}: loss={total_loss/len(loader):.4f}")

    # Unfreeze classifier
    for p in model.classifier.parameters():
        p.requires_grad = True


@torch.no_grad()
def get_embeddings(model, loader, device):
    """
    Extract L2-normalized embeddings from encoder.

    Returns:
        h: (N, d) embeddings
        y: (N,) labels
    """
    model.eval()
    hs, ys = [], []
    for X, y in loader:
        X = X.to(device)
        _, h = model.classify(X)
        hs.append(F.normalize(h, dim=-1).cpu())
        ys.append(y)
    return torch.cat(hs), torch.cat(ys)


def compute_prototypes(h_train, y_train, n_classes):
    """
    Compute class prototypes as L2-normalized mean of training embeddings.

    Returns:
        protos: (K, d) prototypes
    """
    protos = []
    for k in range(n_classes):
        mask = y_train == k
        if mask.sum() == 0:
            # Empty class (shouldn't happen in practice)
            protos.append(torch.zeros(h_train.shape[1]))
        else:
            proto = h_train[mask].mean(0)
            protos.append(F.normalize(proto, dim=0))
    return torch.stack(protos)  # (K, d)


def cosine_dist(h, protos):
    """
    Compute cosine distance: d(h, μ) = 1 - h·μ

    Args:
        h: (N, d) embeddings
        protos: (K, d) prototypes

    Returns:
        dists: (N, K) distances
    """
    return 1.0 - torch.mm(h, protos.T)


def calibrate_tau(h_val, y_val, protos, fpr_target=0.05):
    """
    Calibrate class-conditional thresholds τ_k.
    τ_k = (1 - fpr_target) quantile of intra-class distances on validation set.

    Args:
        h_val: (N, d) validation embeddings
        y_val: (N,) validation labels
        protos: (K, d) prototypes
        fpr_target: target false positive rate (default 0.05 → 95th percentile)

    Returns:
        taus: List[float] of length K
    """
    dists = cosine_dist(h_val, protos)  # (N, K)
    taus = []
    for k in range(protos.shape[0]):
        mask = y_val == k
        if mask.sum() == 0:
            taus.append(float('inf'))
            continue
        d_k = dists[mask, k]
        tau_k = torch.quantile(d_k, 1.0 - fpr_target).item()
        taus.append(tau_k)
    return taus


def predict_open_set(h_test, protos, taus):
    """
    Open-set prediction with prototype-radius rejection.

    Args:
        h_test: (N, d) test embeddings
        protos: (K, d) prototypes
        taus: List[float] of length K

    Returns:
        pred: (N,) predicted labels (-1 = unknown)
        novelty_score: (N,) novelty scores (min cosine distance)
    """
    dists = cosine_dist(h_test, protos)  # (N, K)
    min_dist, nearest = dists.min(dim=1)
    tau_tensor = torch.tensor(taus, dtype=torch.float32)
    tau_nearest = tau_tensor[nearest]
    is_unknown = min_dist > tau_nearest
    pred = nearest.clone()
    pred[is_unknown] = -1  # unknown
    return pred, min_dist  # min_dist = novelty score
