"""
Evaluation methods for all baselines and proposed method.
Includes: MSP, Energy, Mahalanobis, prototype-radius, LOF.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance


def compute_extra_metrics(nov_scores_known, nov_scores_novel, closed_set_correct, closed_set_total):
    """
    Compute FPR95, AUPR, and OSCR from novelty scores.

    Args:
        nov_scores_known: np.ndarray, novelty scores for known-class test samples (higher = more novel)
        nov_scores_novel: np.ndarray, novelty scores for novel-class test samples
        closed_set_correct: int, number of correctly classified known-class samples
        closed_set_total: int, total known-class test samples

    Returns:
        dict with keys: fpr95, aupr, oscr
    """
    # FPR95: FPR on known-class samples when TPR on novel = 95%
    gt = np.concatenate([np.zeros(len(nov_scores_known)), np.ones(len(nov_scores_novel))])
    scores = np.concatenate([nov_scores_known, nov_scores_novel])

    # Sort thresholds descending (higher score = more novel)
    sorted_thresholds = np.sort(scores)[::-1]
    best_fpr95 = 1.0
    for tau in sorted_thresholds:
        tpr = (nov_scores_novel >= tau).mean()
        fpr = (nov_scores_known >= tau).mean()
        if tpr >= 0.95:
            best_fpr95 = fpr
            break

    # AUPR: area under precision-recall curve (novel class = positive)
    aupr = average_precision_score(gt, scores)

    # OSCR: Open-Set Classification Rate
    # = fraction of known samples correctly classified AND not rejected
    # at threshold where novel TPR = 95%
    tau95 = np.quantile(nov_scores_novel, 0.05)  # 95% of novel above this
    known_not_rejected = (nov_scores_known < tau95).sum()
    oscr = (closed_set_correct / closed_set_total) * (known_not_rejected / len(nov_scores_known)) if closed_set_total > 0 else 0.0

    return {"fpr95": float(best_fpr95), "aupr": float(aupr), "oscr": float(oscr)}


@torch.no_grad()
def evaluate_prototype_radius(model, test_k_loader, novel_loader, protos, taus, device):
    """
    Evaluate CE+SupCon+PR (prototype-radius rejection).

    Returns:
        results: dict with keys {auroc, dr, acc, n1}
    """
    from models import get_embeddings, predict_open_set

    model.eval()
    h_test_k, y_test_k = get_embeddings(model, test_k_loader, device)
    h_novel, _ = get_embeddings(model, novel_loader, device)

    # Combine known + novel
    h_all = torch.cat([h_test_k, h_novel])
    gt_novel = torch.cat([
        torch.zeros(len(h_test_k), dtype=torch.long),
        torch.ones(len(h_novel), dtype=torch.long)
    ])

    pred, novelty_score = predict_open_set(h_all, protos, taus)

    # AUROC
    auroc = roc_auc_score(gt_novel.numpy(), novelty_score.numpy())

    # DR: fraction of novel samples correctly flagged as unknown
    novel_pred = pred[len(h_test_k):]
    dr = (novel_pred == -1).float().mean().item()

    # Known-class accuracy (excluding samples predicted as unknown)
    known_pred = pred[:len(h_test_k)]
    known_correct = (known_pred == y_test_k).float()
    known_classified = known_pred != -1
    if known_classified.sum() > 0:
        acc = known_correct[known_classified].mean().item()
    else:
        acc = 0.0

    n1 = 2 * acc * dr / (acc + dr + 1e-8)

    nov_k_np = novelty_score[:len(h_test_k)].numpy()
    nov_n_np = novelty_score[len(h_test_k):].numpy()
    correct_count = int(known_correct[known_classified].sum().item()) if known_classified.sum() > 0 else 0
    extra = compute_extra_metrics(nov_k_np, nov_n_np, correct_count, len(h_test_k))

    return {"auroc": auroc, "dr": dr, "acc": acc, "n1": n1, **extra}


@torch.no_grad()
def evaluate_msp(model, test_k_loader, novel_loader, device):
    """
    Evaluate CE+MSP (Maximum Softmax Probability baseline).

    Returns:
        results: dict with keys {auroc, dr, acc, n1}
    """
    model.eval()

    # Collect MSP scores
    def get_msp(loader):
        scores, labels = [], []
        for X, y in loader:
            X = X.to(device)
            logits, _ = model.classify(X)
            prob = F.softmax(logits, dim=-1)
            msp = prob.max(dim=-1).values
            scores.append(msp.cpu())
            labels.append(y)
        return torch.cat(scores), torch.cat(labels)

    msp_k, yk = get_msp(test_k_loader)
    msp_n, _ = get_msp(novel_loader)

    # Novelty score = 1 - MSP (higher = more novel)
    nov_k = 1.0 - msp_k
    nov_n = 1.0 - msp_n
    nov_all = torch.cat([nov_k, nov_n])
    gt = torch.cat([torch.zeros(len(nov_k)), torch.ones(len(nov_n))])

    # AUROC
    auroc = roc_auc_score(gt.numpy(), nov_all.numpy())

    # Threshold: 95th percentile of known novelty scores
    tau_msp = torch.quantile(nov_k, 0.95).item()
    dr = (nov_n > tau_msp).float().mean().item()

    # Known-class accuracy
    correct, total = 0, 0
    for X, y in test_k_loader:
        X, y = X.to(device), y.to(device)
        logits, _ = model.classify(X)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    acc = correct / total if total > 0 else 0.0

    n1 = 2 * acc * dr / (acc + dr + 1e-8)

    extra = compute_extra_metrics(nov_k.numpy(), nov_n.numpy(), correct, total)

    return {"auroc": auroc, "dr": dr, "acc": acc, "n1": n1, **extra}


@torch.no_grad()
def evaluate_energy(model, test_k_loader, novel_loader, device):
    """
    Evaluate CE+Energy (Liu et al. NeurIPS 2020).
    Energy score = log(sum(exp(logits)))

    Returns:
        results: dict with keys {auroc, dr, acc, n1}
    """
    model.eval()

    def get_energy(loader):
        scores, labels = [], []
        for X, y in loader:
            X = X.to(device)
            logits, _ = model.classify(X)
            energy = torch.logsumexp(logits, dim=-1)
            scores.append(energy.cpu())
            labels.append(y)
        return torch.cat(scores), torch.cat(labels)

    energy_k, yk = get_energy(test_k_loader)
    energy_n, _ = get_energy(novel_loader)

    # Novelty score = -energy (higher = more novel)
    nov_k = -energy_k
    nov_n = -energy_n
    nov_all = torch.cat([nov_k, nov_n])
    gt = torch.cat([torch.zeros(len(nov_k)), torch.ones(len(nov_n))])

    # AUROC
    auroc = roc_auc_score(gt.numpy(), nov_all.numpy())

    # Threshold: 95th percentile
    tau_energy = torch.quantile(nov_k, 0.95).item()
    dr = (nov_n > tau_energy).float().mean().item()

    # Known-class accuracy
    correct, total = 0, 0
    for X, y in test_k_loader:
        X, y = X.to(device), y.to(device)
        logits, _ = model.classify(X)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    acc = correct / total if total > 0 else 0.0

    n1 = 2 * acc * dr / (acc + dr + 1e-8)

    extra = compute_extra_metrics(nov_k.numpy(), nov_n.numpy(), correct, total)

    return {"auroc": auroc, "dr": dr, "acc": acc, "n1": n1, **extra}


@torch.no_grad()
def evaluate_mahalanobis(model, train_loader, val_loader, test_k_loader, novel_loader, device):
    """
    Evaluate CE+Mahalanobis (Lee et al. NeurIPS 2018).
    Class-conditional Gaussian distance in feature space.

    Returns:
        results: dict with keys {auroc, dr, acc, n1}
    """
    from models import get_embeddings

    model.eval()

    # Get training embeddings (not L2-normalized for Mahalanobis)
    h_train_list, y_train_list = [], []
    for X, y in train_loader:
        X = X.to(device)
        _, h = model.classify(X)
        h_train_list.append(h.cpu())
        y_train_list.append(y)
    h_train = torch.cat(h_train_list)
    y_train = torch.cat(y_train_list)

    n_classes = y_train.max().item() + 1

    # Compute class means and shared covariance
    class_means = []
    for k in range(n_classes):
        mask = y_train == k
        class_means.append(h_train[mask].mean(0))
    class_means = torch.stack(class_means)  # (K, d)

    # Shared covariance (pooled)
    centered = []
    for k in range(n_classes):
        mask = y_train == k
        centered.append(h_train[mask] - class_means[k])
    centered = torch.cat(centered)
    cov = EmpiricalCovariance().fit(centered.numpy())
    precision = torch.tensor(cov.precision_, dtype=torch.float32)

    def mahalanobis_dist(h, means, prec):
        """Compute Mahalanobis distance to each class."""
        # h: (N, d), means: (K, d), prec: (d, d)
        # dist[i, k] = (h[i] - means[k])^T @ prec @ (h[i] - means[k])
        dists = []
        for k in range(means.shape[0]):
            diff = h - means[k]  # (N, d)
            dist_k = (diff @ prec * diff).sum(dim=-1)  # (N,)
            dists.append(dist_k)
        return torch.stack(dists, dim=1)  # (N, K)

    # Get test embeddings
    h_test_k, y_test_k = [], []
    for X, y in test_k_loader:
        X = X.to(device)
        _, h = model.classify(X)
        h_test_k.append(h.cpu())
        y_test_k.append(y)
    h_test_k = torch.cat(h_test_k)
    y_test_k = torch.cat(y_test_k)

    h_novel = []
    for X, y in novel_loader:
        X = X.to(device)
        _, h = model.classify(X)
        h_novel.append(h.cpu())
    h_novel = torch.cat(h_novel)

    # Compute distances
    dists_k = mahalanobis_dist(h_test_k, class_means, precision)
    dists_n = mahalanobis_dist(h_novel, class_means, precision)

    # Novelty score = min Mahalanobis distance (higher = more novel)
    nov_k = dists_k.min(dim=1).values
    nov_n = dists_n.min(dim=1).values
    nov_all = torch.cat([nov_k, nov_n])
    gt = torch.cat([torch.zeros(len(nov_k)), torch.ones(len(nov_n))])

    # AUROC
    auroc = roc_auc_score(gt.numpy(), nov_all.numpy())

    # Threshold: 95th percentile
    tau_maha = torch.quantile(nov_k, 0.95).item()
    dr = (nov_n > tau_maha).float().mean().item()

    # Known-class accuracy
    correct, total = 0, 0
    for X, y in test_k_loader:
        X, y = X.to(device), y.to(device)
        logits, _ = model.classify(X)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    acc = correct / total if total > 0 else 0.0

    n1 = 2 * acc * dr / (acc + dr + 1e-8)

    extra = compute_extra_metrics(nov_k.numpy(), nov_n.numpy(), correct, total)

    return {"auroc": auroc, "dr": dr, "acc": acc, "n1": n1, **extra}


@torch.no_grad()
def evaluate_lof(model, train_loader, test_k_loader, novel_loader, device, n_neighbors=20):
    """
    Evaluate CE+SupCon+LOF (Local Outlier Factor).

    Returns:
        results: dict with keys {auroc, dr, acc, n1}
    """
    from models import get_embeddings

    model.eval()

    # Get embeddings
    h_train, y_train = get_embeddings(model, train_loader, device)
    h_test_k, y_test_k = get_embeddings(model, test_k_loader, device)
    h_novel, _ = get_embeddings(model, novel_loader, device)

    # Fit LOF on training set
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(h_train.numpy())

    # Predict novelty scores (negative_outlier_factor)
    nov_k = -lof.score_samples(h_test_k.numpy())
    nov_n = -lof.score_samples(h_novel.numpy())
    nov_all = np.concatenate([nov_k, nov_n])
    gt = np.concatenate([np.zeros(len(nov_k)), np.ones(len(nov_n))])

    # AUROC
    auroc = roc_auc_score(gt, nov_all)

    # Threshold: 95th percentile
    tau_lof = np.quantile(nov_k, 0.95)
    dr = (nov_n > tau_lof).mean()

    # Known-class accuracy
    correct, total = 0, 0
    for X, y in test_k_loader:
        X, y = X.to(device), y.to(device)
        logits, _ = model.classify(X)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    acc = correct / total if total > 0 else 0.0

    n1 = 2 * acc * dr / (acc + dr + 1e-8)

    extra = compute_extra_metrics(nov_k, nov_n, correct, total)

    return {"auroc": auroc, "dr": float(dr), "acc": acc, "n1": n1, **extra}


def print_results(results, tag):
    """Pretty-print evaluation results."""
    print(f"\n[{tag}]")
    print(f"  AUROC : {results['auroc']:.4f}")
    print(f"  DR    : {results['dr']:.4f}")
    print(f"  Acc   : {results['acc']:.4f}")
    print(f"  N1    : {results['n1']:.4f}")
    if "fpr95" in results:
        print(f"  FPR95 : {results['fpr95']:.4f}")
        print(f"  AUPR  : {results['aupr']:.4f}")
        print(f"  OSCR  : {results['oscr']:.4f}")
