"""
nNFST: Nearest-class Null-space Feature Subspace Transform (no novel prior).

Adaptation of NFST for open-set recognition without novel-class prior.
Only known-class training embeddings are used to build per-class subspaces.
Novelty score = minimum reconstruction error across all known-class subspaces.

Reference: Zhang et al., "Towards Open Set Deep Networks" (adapted).
Adaptation: remove novel-class null-space; use known-class within-class scatter
            subspace only. Test sample is novel if its min reconstruction error
            across all K subspaces exceeds a calibrated threshold.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from evaluate import compute_extra_metrics


class NNFSTClassifier:
    """
    No-prior NFST classifier.

    For each known class k, compute the within-class subspace via PCA on
    centered class embeddings. Novelty score for a test point x is:

        score(x) = min_k  ||x - proj_k(x)||^2

    where proj_k(x) is the projection of x onto class k's subspace.
    Lower score = more likely known; higher score = more likely novel.

    Threshold tau_k is calibrated on the validation set at the 95th percentile
    of known-class reconstruction errors (zero novel samples used).
    """

    def __init__(self, n_components=None, var_ratio=0.95):
        """
        Args:
            n_components: fixed number of PCA components per class.
                          If None, use var_ratio to select automatically.
            var_ratio: cumulative variance ratio to retain (used when n_components=None).
        """
        self.n_components = n_components
        self.var_ratio = var_ratio
        self.subspaces = []   # List of (mean_k, V_k) per class
        self.n_classes = 0

    def fit(self, h_train: np.ndarray, y_train: np.ndarray):
        """
        Build per-class PCA subspaces from training embeddings.

        Args:
            h_train: (N, d) L2-normalized embeddings
            y_train: (N,) integer class labels 0..K-1
        """
        self.n_classes = int(y_train.max()) + 1
        self.subspaces = []

        for k in range(self.n_classes):
            mask = y_train == k
            H_k = h_train[mask]  # (N_k, d)

            if len(H_k) < 2:
                # Degenerate: store identity subspace
                self.subspaces.append((np.zeros(h_train.shape[1]), None))
                continue

            mean_k = H_k.mean(axis=0)
            H_centered = H_k - mean_k  # (N_k, d)

            # SVD for PCA
            U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
            # Vt: (min(N_k,d), d) — rows are principal components

            if self.n_components is not None:
                n_comp = min(self.n_components, Vt.shape[0])
            else:
                # Select components by cumulative variance
                var = S ** 2
                cum_var = np.cumsum(var) / (var.sum() + 1e-12)
                n_comp = int(np.searchsorted(cum_var, self.var_ratio)) + 1
                n_comp = min(n_comp, Vt.shape[0])

            V_k = Vt[:n_comp]  # (n_comp, d) — top principal components

            self.subspaces.append((mean_k, V_k))

    def reconstruction_error(self, h: np.ndarray) -> np.ndarray:
        """
        Compute per-class reconstruction errors.

        Args:
            h: (N, d) embeddings

        Returns:
            errors: (N, K) reconstruction errors per class
        """
        N, d = h.shape
        errors = np.zeros((N, self.n_classes))

        for k, (mean_k, V_k) in enumerate(self.subspaces):
            if V_k is None:
                # Degenerate class: use full distance from mean
                errors[:, k] = np.sum((h - mean_k) ** 2, axis=1)
                continue

            h_centered = h - mean_k  # (N, d)
            # Project onto subspace: proj = h_centered @ V_k.T @ V_k
            coords = h_centered @ V_k.T  # (N, n_comp)
            proj = coords @ V_k          # (N, d)
            residual = h_centered - proj  # (N, d)
            errors[:, k] = np.sum(residual ** 2, axis=1)

        return errors

    def novelty_score(self, h: np.ndarray) -> np.ndarray:
        """
        Novelty score = min reconstruction error across all known classes.

        Args:
            h: (N, d) embeddings

        Returns:
            scores: (N,) novelty scores (higher = more novel)
        """
        errors = self.reconstruction_error(h)
        return errors.min(axis=1)

    def predict_class(self, h: np.ndarray) -> np.ndarray:
        """
        Closed-set prediction: nearest class by reconstruction error.

        Args:
            h: (N, d) embeddings

        Returns:
            pred: (N,) predicted class indices
        """
        errors = self.reconstruction_error(h)
        return errors.argmin(axis=1)

    def calibrate_threshold(self, h_val: np.ndarray, y_val: np.ndarray,
                            fpr_target: float = 0.05) -> float:
        """
        Calibrate global threshold at (1 - fpr_target) quantile of known-class
        novelty scores on validation set. Zero novel samples used.

        Args:
            h_val: (N, d) validation embeddings
            y_val: (N,) validation labels (all known classes)
            fpr_target: target FPR (default 0.05 → 95th percentile)

        Returns:
            tau: float threshold
        """
        scores = self.novelty_score(h_val)
        tau = float(np.quantile(scores, 1.0 - fpr_target))
        return tau


def evaluate_nnfst(model, train_loader, val_loader, test_k_loader, novel_loader,
                   device, n_components=None, var_ratio=0.95):
    """
    Full nNFST evaluation pipeline.

    Steps:
    1. Extract L2-normalized embeddings from encoder (no SupCon training needed)
    2. Fit per-class PCA subspaces on training embeddings
    3. Calibrate threshold on validation set
    4. Evaluate on test set (known + novel)

    Args:
        model: trained BaselineModel or SepSpaceModel (CE-only Stage 1)
        train_loader, val_loader, test_k_loader, novel_loader: DataLoaders
        device: torch device
        n_components: PCA components per class (None = auto by var_ratio)
        var_ratio: cumulative variance to retain

    Returns:
        results: dict {auroc, dr, acc, n1}
    """
    from models import get_embeddings

    model.eval()

    # Extract embeddings
    h_train, y_train = get_embeddings(model, train_loader, device)
    h_val, y_val = get_embeddings(model, val_loader, device)
    h_test_k, y_test_k = get_embeddings(model, test_k_loader, device)
    h_novel, _ = get_embeddings(model, novel_loader, device)

    h_train_np = h_train.numpy()
    y_train_np = y_train.numpy()
    h_val_np = h_val.numpy()
    y_val_np = y_val.numpy()
    h_test_k_np = h_test_k.numpy()
    h_novel_np = h_novel.numpy()

    # Fit nNFST
    clf = NNFSTClassifier(n_components=n_components, var_ratio=var_ratio)
    clf.fit(h_train_np, y_train_np)

    # Calibrate threshold
    tau = clf.calibrate_threshold(h_val_np, y_val_np, fpr_target=0.05)

    # Novelty scores on test set
    nov_k = clf.novelty_score(h_test_k_np)
    nov_n = clf.novelty_score(h_novel_np)

    nov_all = np.concatenate([nov_k, nov_n])
    gt = np.concatenate([np.zeros(len(nov_k)), np.ones(len(nov_n))])

    # AUROC
    auroc = roc_auc_score(gt, nov_all)

    # DR: fraction of novel samples with score > tau
    dr = float((nov_n > tau).mean())

    # Known-class accuracy (closed-set prediction on test_k)
    pred_k = clf.predict_class(h_test_k_np)
    correct = int((pred_k == y_test_k.numpy()).sum())
    total = len(y_test_k)
    acc = correct / total

    n1 = 2 * acc * dr / (acc + dr + 1e-8)

    extra = compute_extra_metrics(nov_k, nov_n, correct, total)

    return {"auroc": auroc, "dr": dr, "acc": acc, "n1": n1, **extra}