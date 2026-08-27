"""
Main experiment runner for SepSpace.
Supports all datasets, methods, and class-holdout protocol.
"""

import os
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_loader import load_ustc_tfc2016, load_mcfp, load_iscx, split_data
from models import (
    SepSpaceModel, BaselineModel,
    train_stage1, train_stage2,
    get_embeddings, compute_prototypes, calibrate_tau
)
from evaluate import (
    evaluate_prototype_radius, evaluate_msp, evaluate_energy,
    evaluate_mahalanobis, evaluate_lof, print_results
)
from nnfst import evaluate_nnfst


class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(args):
    """Run a single class-holdout experiment."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print(f"\n{'='*72}")
    print(f"Dataset: {args.dataset}, Novel class: {args.novel_class}")
    print(f"Method: {args.method}, N_cap: {args.n_cap}")
    print(f"{'='*72}")

    if args.dataset == "ustc":
        X, y, class_names, vocab_size = load_ustc_tfc2016(
            args.data_root, args.n_cap, seed=args.seed
        )
    elif args.dataset == "mcfp":
        X, y, class_names, vocab_size = load_mcfp(
            args.data_root, args.n_cap, seed=args.seed
        )
    elif args.dataset.startswith("iscx"):
        _subset_map = {
            "iscx-vpn-app":     ("VPN",    "app"),
            "iscx-vpn-service": ("VPN",    "service"),
            "iscx-nonvpn-app":     ("nonVPN", "app"),
            "iscx-nonvpn-service": ("nonVPN", "service"),
            # legacy aliases kept for backward compat
            "iscx-vpn":    ("VPN",    "app"),
            "iscx-nonvpn": ("nonVPN", "service"),
        }
        subset, label_scheme = _subset_map[args.dataset]
        X, y, class_names, vocab_size = load_iscx(
            args.data_root, subset, label_scheme, args.n_cap, seed=args.seed
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Loaded {len(y)} samples, {len(class_names)} classes")
    print(f"Classes: {class_names}")

    # Find novel class index
    if args.novel_class.isdigit():
        novel_idx = int(args.novel_class)
    else:
        if args.novel_class not in class_names:
            raise ValueError(f"Novel class '{args.novel_class}' not in {class_names}")
        novel_idx = class_names.index(args.novel_class)

    novel_name = class_names[novel_idx]
    print(f"Novel class: {novel_name} (idx={novel_idx})")

    # Split data
    (X_train, y_train, X_val, y_val,
     X_test_k, y_test_k, X_novel, y_novel, n_classes) = split_data(
        X, y, novel_idx, seed=args.seed
    )

    print(f"Split: train={len(y_train)}, val={len(y_val)}, "
          f"test_known={len(y_test_k)}, test_novel={len(y_novel)}")

    # Create dataloaders
    train_ds = TrafficDataset(X_train, y_train)
    val_ds = TrafficDataset(X_val, y_val)
    test_k_ds = TrafficDataset(X_test_k, y_test_k)
    novel_ds = TrafficDataset(X_novel, np.zeros(len(y_novel), dtype=np.int64))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_k_loader = DataLoader(test_k_ds, batch_size=args.batch_size)
    novel_loader = DataLoader(novel_ds, batch_size=args.batch_size)

    # Train and evaluate based on method
    results = {}

    if args.method == "ce_msp":
        print("\n[Training CE+MSP baseline]")
        model = BaselineModel(vocab_size, n_classes, d_model=args.d_model).to(device)
        train_stage1(model, train_loader, args.epochs_ce, args.lr, device)
        results = evaluate_msp(model, test_k_loader, novel_loader, device)
        print_results(results, f"CE+MSP | novel={novel_name}")

    elif args.method == "ce_energy":
        print("\n[Training CE+Energy baseline]")
        model = BaselineModel(vocab_size, n_classes, d_model=args.d_model).to(device)
        train_stage1(model, train_loader, args.epochs_ce, args.lr, device)
        results = evaluate_energy(model, test_k_loader, novel_loader, device)
        print_results(results, f"CE+Energy | novel={novel_name}")

    elif args.method == "ce_mahalanobis":
        print("\n[Training CE+Mahalanobis baseline]")
        model = BaselineModel(vocab_size, n_classes, d_model=args.d_model).to(device)
        train_stage1(model, train_loader, args.epochs_ce, args.lr, device)
        results = evaluate_mahalanobis(
            model, train_loader, val_loader, test_k_loader, novel_loader, device
        )
        print_results(results, f"CE+Mahalanobis | novel={novel_name}")

    elif args.method == "ce_pr":
        print("\n[Training CE+PR (no SupCon)]")
        model = SepSpaceModel(vocab_size, n_classes, d_model=args.d_model, d_proj=args.d_proj).to(device)
        train_stage1(model, train_loader, args.epochs_ce, args.lr, device)

        h_train, y_tr = get_embeddings(model, train_loader, device)
        h_val, y_vl = get_embeddings(model, val_loader, device)
        protos = compute_prototypes(h_train, y_tr, n_classes)
        taus = calibrate_tau(h_val, y_vl, protos, fpr_target=0.05)

        results = evaluate_prototype_radius(
            model, test_k_loader, novel_loader, protos, taus, device
        )
        print_results(results, f"CE+PR | novel={novel_name}")

    elif args.method == "sepspace":
        print("\n[Training SepSpace (CE+SupCon+PR)]")
        model = SepSpaceModel(vocab_size, n_classes, d_model=args.d_model, d_proj=args.d_proj).to(device)
        train_stage1(model, train_loader, args.epochs_ce, args.lr, device)
        train_stage2(model, train_loader, args.epochs_joint, args.lr * 0.3, args.lambda_sup, device)

        h_train, y_tr = get_embeddings(model, train_loader, device)
        h_val, y_vl = get_embeddings(model, val_loader, device)
        protos = compute_prototypes(h_train, y_tr, n_classes)
        taus = calibrate_tau(h_val, y_vl, protos, fpr_target=0.05)

        results = evaluate_prototype_radius(
            model, test_k_loader, novel_loader, protos, taus, device
        )
        print_results(results, f"SepSpace | novel={novel_name}")

    elif args.method == "sepspace_frozen":
        print("\n[Training SepSpace (frozen encoder)]")
        model = SepSpaceModel(vocab_size, n_classes, d_model=args.d_model, d_proj=args.d_proj).to(device)
        train_stage1(model, train_loader, args.epochs_ce, args.lr, device)

        # Stage 2 with frozen encoder
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = False

        opt = torch.optim.Adam(model.proj_head.parameters(), lr=args.lr * 0.3)
        model.train()
        for ep in range(args.epochs_joint):
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                z = model.project(X)
                from models import supcon_loss
                loss = supcon_loss(z, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if (ep + 1) % 5 == 0:
                print(f"  Stage2 (frozen) ep{ep+1:02d}: loss={total_loss/len(train_loader):.4f}")

        # Unfreeze
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

        h_train, y_tr = get_embeddings(model, train_loader, device)
        h_val, y_vl = get_embeddings(model, val_loader, device)
        protos = compute_prototypes(h_train, y_tr, n_classes)
        taus = calibrate_tau(h_val, y_vl, protos, fpr_target=0.05)

        results = evaluate_prototype_radius(
            model, test_k_loader, novel_loader, protos, taus, device
        )
        print_results(results, f"SepSpace (frozen) | novel={novel_name}")

    elif args.method == "sepspace_lof":
        print("\n[Training SepSpace+LOF]")
        model = SepSpaceModel(vocab_size, n_classes, d_model=args.d_model, d_proj=args.d_proj).to(device)
        train_stage1(model, train_loader, args.epochs_ce, args.lr, device)
        train_stage2(model, train_loader, args.epochs_joint, args.lr * 0.3, args.lambda_sup, device)

        results = evaluate_lof(model, train_loader, test_k_loader, novel_loader, device)
        print_results(results, f"SepSpace+LOF | novel={novel_name}")

    elif args.method == "nnfst":
        print("\n[Training nNFST (CE Stage1 + subspace OSR)]")
        model = BaselineModel(vocab_size, n_classes, d_model=args.d_model).to(device)
        train_stage1(model, train_loader, args.epochs_ce, args.lr, device)
        results = evaluate_nnfst(
            model, train_loader, val_loader, test_k_loader, novel_loader,
            device, n_components=args.nnfst_components, var_ratio=args.nnfst_var_ratio
        )
        print_results(results, f"nNFST | novel={novel_name}")

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save results
    output = {
        "dataset": args.dataset,
        "method": args.method,
        "novel_class": novel_name,
        "novel_idx": novel_idx,
        "n_cap": args.n_cap,
        "seed": args.seed,
        "results": results,
        "config": vars(args)
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.method}_{novel_name}_seed{args.seed}.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SepSpace Experiment Runner")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ustc", "mcfp",
                                 "iscx-vpn-app", "iscx-vpn-service",
                                 "iscx-nonvpn-app", "iscx-nonvpn-service",
                                 "iscx-vpn", "iscx-nonvpn"],  # legacy aliases
                        help="Dataset name")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--novel_class", type=str, required=True,
                        help="Novel class name or index")
    parser.add_argument("--n_cap", type=int, default=2000,
                        help="Max samples per class")

    # Method
    parser.add_argument("--method", type=str, required=True,
                        choices=["ce_msp", "ce_energy", "ce_mahalanobis",
                                 "ce_pr", "sepspace", "sepspace_frozen", "sepspace_lof",
                                 "nnfst"],
                        help="Method to evaluate")

    # Model
    parser.add_argument("--d_model", type=int, default=64,
                        help="Encoder embedding dimension")
    parser.add_argument("--d_proj", type=int, default=32,
                        help="Projection head dimension")

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs_ce", type=int, default=20,
                        help="Stage 1 epochs")
    parser.add_argument("--epochs_joint", type=int, default=15,
                        help="Stage 2 epochs")
    parser.add_argument("--lambda_sup", type=float, default=0.5,
                        help="SupCon loss weight")

    # nNFST-specific
    parser.add_argument("--nnfst_components", type=int, default=None,
                        help="Fixed PCA components per class for nNFST (None = auto)")
    parser.add_argument("--nnfst_var_ratio", type=float, default=0.95,
                        help="Cumulative variance ratio for nNFST PCA (default 0.95)")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
