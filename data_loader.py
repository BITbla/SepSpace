"""
Data loading and preprocessing for SepSpace experiments.
Supports: USTC-TFC2016, MCFP, ISCX-VPN, ISCX-nonVPN.
"""

import os
import json
import glob
import random
import numpy as np
from scapy.all import rdpcap, IP
from collections import defaultdict

CACHE_ROOT = os.path.join("data", "preprocessed")


def pcap_to_units(pcap_path, max_units, window_l=8, stride_s=4, mtu=1500):
    """
    Extract Channel Units from a single PCAP file.

    Args:
        pcap_path: Path to PCAP file
        max_units: Maximum number of Channel Units to extract
        window_l: Channel Unit length (number of packets)
        stride_s: Sliding window stride
        mtu: Maximum transmission unit for length clipping

    Returns:
        List of Channel Units, each as a list of token IDs
    """
    try:
        pkts = rdpcap(pcap_path)
    except Exception as e:
        print(f"  [WARN] Cannot read {pcap_path}: {e}")
        return []

    # Build (direction, length) sequence
    seq = []
    first_src = None
    for pkt in pkts:
        if not pkt.haslayer(IP):
            continue
        src = pkt[IP].src
        if first_src is None:
            first_src = src
        direction = 0 if pkt[IP].src == first_src else 1
        length = min(len(pkt), mtu)
        seq.append((direction, length))
        if len(seq) >= max_units * window_l * 2:
            break

    if len(seq) < window_l:
        return []

    # Sliding window → token IDs
    N_RESERVED = 2  # 0=PAD, 1=CLS
    units = []
    i = 0
    while i + window_l <= len(seq) and len(units) < max_units:
        window = seq[i : i + window_l]
        tokens = [N_RESERVED + length + direction * mtu for direction, length in window]
        units.append(tokens)
        i += stride_s

    return units


def load_ustc_tfc2016(root, n_cap, window_l=8, stride_s=4, mtu=1500, seed=42):
    """
    Load USTC-TFC2016 dataset with preprocessing cache.

    Returns:
        X: np.array of shape (N, window_l), dtype=int32
        y: np.array of shape (N,), dtype=int64
        class_names: List[str]
        vocab_size: int
    """
    # Check cache
    cache_dir = os.path.join(CACHE_ROOT, "ustc")
    cache_path = os.path.join(cache_dir, "features.npy")
    meta_path = os.path.join(cache_dir, "meta.json")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Verify preprocessing parameters match (root path and seed NOT required for cache hit)
        # seed only affects train/val/test split, not the data content itself.
        if (
            meta.get("n_cap") == n_cap
            and meta.get("window_l") == window_l
            and meta.get("stride_s") == stride_s
            and meta.get("mtu") == mtu
        ):

            print(f"Loading from cache: {cache_dir}")
            X = np.load(os.path.join(cache_dir, "features.npy"))
            y = np.load(os.path.join(cache_dir, "labels.npy"))

            with open(os.path.join(cache_dir, "class_stats.json"), "r") as f:
                stats = json.load(f)

            return X, y, stats["class_names"], stats["vocab_size"]

    # No cache, do full preprocessing
    malware_dir = os.path.join(root, "Malware")
    benign_dir = os.path.join(root, "Benign")

    # Exclude adware adload and wannaCry (extra classes not in original USTC)
    exclude_classes = ["adware adload", "wannaCry"]

    # Merge split files: SMB-1/SMB-2 → SMB, Weibo-1/2/3/4 → Weibo
    merge_map = {
        "SMB-1": "SMB",
        "SMB-2": "SMB",
        "Weibo-1": "Weibo",
        "Weibo-2": "Weibo",
        "Weibo-3": "Weibo",
        "Weibo-4": "Weibo",
    }

    class_files = []
    for pcap in sorted(glob.glob(os.path.join(malware_dir, "*.pcap"))):
        name = os.path.basename(pcap).replace(".pcap", "")
        if name in exclude_classes:
            print(f"  [EXCLUDE] M-{name}: extra class not in original USTC")
            continue
        class_files.append((pcap, f"M-{name}"))
    for pcap in sorted(glob.glob(os.path.join(benign_dir, "*.pcap"))):
        name = os.path.basename(pcap).replace(".pcap", "")
        if name in exclude_classes:
            print(f"  [EXCLUDE] B-{name}: extra class not in original USTC")
            continue
        class_files.append((pcap, f"B-{name}"))

    print(f"Loading USTC-TFC2016 from {root}")

    # Aggregate units by merged class name
    class_units = defaultdict(list)
    for pcap_path, class_name in class_files:
        # Apply merge mapping
        base_name = class_name.split("-", 1)[1]  # Remove M-/B- prefix
        merged_name = merge_map.get(base_name, base_name)
        merged_class = f"{class_name[0]}-{merged_name}"  # Restore M-/B- prefix

        units = pcap_to_units(
            pcap_path,
            max_units=n_cap * 2,
            window_l=window_l,
            stride_s=stride_s,
            mtu=mtu,
        )
        class_units[merged_class].extend(units)
        print(f"  {class_name} → {merged_class}: {len(units)} units")

    # Sample from merged classes
    all_units, all_labels, class_names = [], [], []
    for label_idx, (class_name, units) in enumerate(sorted(class_units.items())):
        n = min(len(units), n_cap)
        if n < 200:
            print(f"  [SKIP] {class_name}: only {n} units (< 200 threshold)")
            continue

        # Random sample if needed
        if len(units) > n_cap:
            random.seed(seed)
            units = random.sample(units, n_cap)
        else:
            units = units[:n]

        print(f"  {class_name}: {n} units (final)")
        all_units.extend(units)
        all_labels.extend([label_idx] * n)
        class_names.append(class_name)

    # Pad to window_l
    vocab_size = 2 + 2 * mtu + 1
    padded = []
    for u in all_units:
        if len(u) < window_l:
            u = u + [0] * (window_l - len(u))
        padded.append(u[:window_l])

    X = np.array(padded, dtype=np.int32)
    y = np.array(all_labels, dtype=np.int64)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "features.npy"), X)
    np.save(os.path.join(cache_dir, "labels.npy"), y)

    with open(os.path.join(cache_dir, "class_stats.json"), "w") as f:
        json.dump(
            {
                "class_names": class_names,
                "vocab_size": vocab_size,
                "class_counts": {
                    name: int(np.sum(y == i)) for i, name in enumerate(class_names)
                },
            },
            f,
            indent=2,
        )

    with open(meta_path, "w") as f:
        json.dump(
            {
                "n_cap": n_cap,
                "window_l": window_l,
                "stride_s": stride_s,
                "mtu": mtu,
                "seed": seed,
                "root": root,
                "timestamp": str(np.datetime64("now")),
            },
            f,
            indent=2,
        )

    print(f"Saved to cache: {cache_dir}")

    return X, y, class_names, vocab_size


def load_mcfp(root, n_cap, window_l=8, stride_s=4, mtu=1500, seed=42):
    """
    Load MCFP dataset (8 malware families) with preprocessing cache.

    Returns:
        X, y, class_names, vocab_size
    """
    # Check cache
    cache_dir = os.path.join(CACHE_ROOT, "mcfp")
    cache_path = os.path.join(cache_dir, "features.npy")
    meta_path = os.path.join(cache_dir, "meta.json")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Verify preprocessing parameters match (root path and seed NOT required for cache hit)
        # seed only affects train/val/test split, not the data content itself.
        if (
            meta.get("n_cap") == n_cap
            and meta.get("window_l") == window_l
            and meta.get("stride_s") == stride_s
            and meta.get("mtu") == mtu
        ):

            print(f"Loading from cache: {cache_dir}")
            X = np.load(os.path.join(cache_dir, "features.npy"))
            y = np.load(os.path.join(cache_dir, "labels.npy"))

            with open(os.path.join(cache_dir, "class_stats.json"), "r") as f:
                stats = json.load(f)

            return X, y, stats["class_names"], stats["vocab_size"]

    # No cache, do full preprocessing
    families = [
        "Cobalt",
        "Dridex",
        "Normal-32",
        "TrickBot",
        "Trojan_Downloader",
        "Trojan_Locky",
        "Worm_Allape",
        "Zeus",
    ]

    print(f"Loading MCFP from {root}")
    all_units, all_labels, class_names = [], [], []

    for label_idx, family in enumerate(families):
        family_dir = os.path.join(root, family)
        if not os.path.isdir(family_dir):
            print(f"  [SKIP] {family}: directory not found")
            continue

        # Aggregate all PCAPs in this family
        pcaps = glob.glob(os.path.join(family_dir, "*.pcap")) + glob.glob(
            os.path.join(family_dir, "*.pcapng")
        )

        family_units = []
        for pcap in pcaps:
            units = pcap_to_units(
                pcap, max_units=n_cap * 2, window_l=window_l, stride_s=stride_s, mtu=mtu
            )
            family_units.extend(units)
            if len(family_units) >= n_cap * 2:
                break

        n = min(len(family_units), n_cap)
        if n < 200:
            print(f"  [SKIP] {family}: only {n} units (< 200 threshold)")
            continue

        # Random sample
        if len(family_units) > n_cap:
            random.seed(seed)
            family_units = random.sample(family_units, n_cap)
        else:
            family_units = family_units[:n]

        print(f"  {family}: {n} units")
        all_units.extend(family_units)
        all_labels.extend([label_idx] * n)
        class_names.append(family)

    # Pad
    vocab_size = 2 + 2 * mtu + 1
    padded = []
    for u in all_units:
        if len(u) < window_l:
            u = u + [0] * (window_l - len(u))
        padded.append(u[:window_l])

    X = np.array(padded, dtype=np.int32)
    y = np.array(all_labels, dtype=np.int64)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "features.npy"), X)
    np.save(os.path.join(cache_dir, "labels.npy"), y)

    with open(os.path.join(cache_dir, "class_stats.json"), "w") as f:
        json.dump(
            {
                "class_names": class_names,
                "vocab_size": vocab_size,
                "class_counts": {
                    name: int(np.sum(y == i)) for i, name in enumerate(class_names)
                },
            },
            f,
            indent=2,
        )

    with open(meta_path, "w") as f:
        json.dump(
            {
                "n_cap": n_cap,
                "window_l": window_l,
                "stride_s": stride_s,
                "mtu": mtu,
                "seed": seed,
                "root": root,
                "timestamp": str(np.datetime64("now")),
            },
            f,
            indent=2,
        )

    print(f"Saved to cache: {cache_dir}")

    return X, y, class_names, vocab_size


# ---------------------------------------------------------------------------
# ISCX label functions — one per (subset, label_scheme) combination
# Returns None to skip uncertain files.
# ---------------------------------------------------------------------------


def _vpn_app(basename):
    """VPN → application-level label."""
    b = basename.replace("vpn_", "").replace(".pcap", "").replace(".pcapng", "")
    if b.startswith("aim"):
        return "aim"
    if b.startswith("facebook"):
        return "facebook"
    if b.startswith("hangouts"):
        return "hangouts"
    if b.startswith("icq"):
        return "icq"
    if b.startswith("netflix"):
        return "netflix"
    if b.startswith("skype"):
        return "skype"
    if b.startswith("youtube"):
        return "youtube"
    if b.startswith("spotify"):
        return "spotify"
    if b.startswith("vimeo"):
        return "vimeo"
    if b.startswith("ftps"):
        return "ftps"
    if b.startswith("sftp"):
        return "sftp"
    if b.startswith("email"):
        return "email"
    if b.startswith("voipbuster"):
        return "voipbuster"
    if b.startswith("bittorrent"):
        return "bittorrent"
    return None


def _vpn_service(basename):
    """VPN → service-level label."""
    b = basename.replace("vpn_", "").replace(".pcap", "").replace(".pcapng", "")
    if b.startswith("email"):
        return "email"
    if b.startswith("aim_chat"):
        return "chat"
    if b.startswith("facebook_chat"):
        return "chat"
    if b.startswith("hangouts_chat"):
        return "chat"
    if b.startswith("icq_chat"):
        return "chat"
    if (
        b.startswith("netflix")
        or b.startswith("youtube")
        or b.startswith("vimeo")
        or b.startswith("spotify")
    ):
        return "streaming"
    if b.startswith("ftps") or b.startswith("sftp"):
        return "file_transfer"
    if (
        b.startswith("facebook_audio")
        or b.startswith("hangouts_audio")
        or b.startswith("skype_audio")
        or b.startswith("voipbuster")
    ):
        return "voip"
    if b.startswith("bittorrent"):
        return "p2p"
    return None  # skip: skype_chat, skype_files (mixed/uncertain)


def _nonvpn_app(basename):
    """NonVPN → application-level label."""
    b = basename.replace(".pcap", "").replace(".pcapng", "")
    if b.startswith("aimchat") or b.startswith("aim_chat"):
        return "aim"
    if b.startswith("email"):
        return "email"
    if b.startswith("facebook"):
        return "facebook"
    if b.startswith("hangouts") or b.startswith("hangout_"):
        return "hangouts"
    if b.startswith("icqchat") or b.startswith("icq_chat"):
        return "icq"
    if b.startswith("netflix"):
        return "netflix"
    if b.startswith("skype"):
        return "skype"
    if b.startswith("youtube"):
        return "youtube"
    if b.startswith("spotify"):
        return "spotify"
    if b.startswith("vimeo"):
        return "vimeo"
    if b.startswith("voipbuster"):
        return "voipbuster"
    if b.startswith("ftps"):
        return "ftps"
    if (
        b.startswith("sftp")
        or b.startswith("sftpdown")
        or b.startswith("sftpup")
        or b.startswith("sftp_")
    ):
        return "sftp"
    if b.startswith("scp") or b.startswith("scpdown") or b.startswith("scpup"):
        return "scp"
    return None  # skip: gmailchat* (uncertain)


def _nonvpn_service(basename):
    """NonVPN → service-level label (per official ISCX annotation)."""
    b = basename.replace(".pcap", "").replace(".pcapng", "")
    if b.startswith("email"):
        return "email"
    if b.startswith("aimchat") or b.startswith("aim_chat"):
        return "chat"
    if b.startswith("icqchat") or b.startswith("icq_chat"):
        return "chat"
    if b.startswith("facebookchat") or b.startswith("facebook_chat"):
        return "chat"
    if b.startswith("hangout_chat") or b.startswith("hangouts_chat"):
        return "chat"
    if b.startswith("netflix"):
        return "streaming"
    if b.startswith("youtube") and not b.startswith("youtubehtml"):
        return "streaming"
    if b.startswith("ftps"):
        return "file_transfer"
    if (
        b.startswith("sftp")
        or b.startswith("sftpdown")
        or b.startswith("sftpup")
        or b.startswith("sftp_")
    ):
        return "file_transfer"
    if b.startswith("scp") or b.startswith("scpdown") or b.startswith("scpup"):
        return "file_transfer"
    if b.startswith("facebook_audio"):
        return "voip"
    if b.startswith("hangouts_audio"):
        return "voip"
    return None  # skip: facebook_video*, gmailchat*, youtubehtml5*, etc.


_ISCX_LABEL_FN = {
    ("VPN", "app"): _vpn_app,
    ("VPN", "service"): _vpn_service,
    ("nonVPN", "app"): _nonvpn_app,
    ("nonVPN", "service"): _nonvpn_service,
}


def load_iscx(
    root,
    subset="nonVPN",
    label_scheme="app",
    n_cap=5000,
    window_l=8,
    stride_s=4,
    mtu=1500,
    seed=42,
):
    """
    Load ISCX VPN-nonVPN 2016 dataset with preprocessing cache.

    Four independent datasets are supported via (subset, label_scheme):
      ("VPN",    "app")     → iscx_vpn_app/
      ("VPN",    "service") → iscx_vpn_service/
      ("nonVPN", "app")     → iscx_nonvpn_app/
      ("nonVPN", "service") → iscx_nonvpn_service/

    Files that cannot be confidently assigned to a class are skipped.

    Args:
        subset:       "VPN" or "nonVPN"
        label_scheme: "app" or "service"

    Returns:
        X, y, class_names, vocab_size
    """
    assert subset in (
        "VPN",
        "nonVPN",
    ), f"subset must be 'VPN' or 'nonVPN', got {subset!r}"
    assert label_scheme in (
        "app",
        "service",
    ), f"label_scheme must be 'app' or 'service', got {label_scheme!r}"

    cache_key = f"iscx_{subset.lower()}_{label_scheme}"
    cache_dir = os.path.join(CACHE_ROOT, cache_key)
    meta_path = os.path.join(cache_dir, "meta.json")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if (
            meta.get("n_cap") == n_cap
            and meta.get("window_l") == window_l
            and meta.get("stride_s") == stride_s
            and meta.get("mtu") == mtu
            and meta.get("label_scheme") == label_scheme
        ):
            print(f"Loading from cache: {cache_dir}")
            X = np.load(os.path.join(cache_dir, "features.npy"))
            y = np.load(os.path.join(cache_dir, "labels.npy"))
            with open(os.path.join(cache_dir, "class_stats.json"), "r") as f:
                stats = json.load(f)
            return X, y, stats["class_names"], stats["vocab_size"]

    if subset == "VPN":
        dirs = [os.path.join(root, "VPN-PCAPS-01"), os.path.join(root, "VPN-PCAPs-02")]
    else:
        dirs = [
            os.path.join(root, d)
            for d in ["NonVPN-PCAPs-01", "NonVPN-PCAPs-02", "NonVPN-PCAPs-03"]
        ]

    label_fn = _ISCX_LABEL_FN[(subset, label_scheme)]
    print(f"Loading ISCX-{subset} ({label_scheme}-level) from {root}")

    # Aggregate PCAPs by class label
    class_pcaps = defaultdict(list)
    skipped = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for pcap in sorted(
            glob.glob(os.path.join(d, "*.pcap"))
            + glob.glob(os.path.join(d, "*.pcapng"))
        ):
            label = label_fn(os.path.basename(pcap).lower())
            if label is None:
                skipped.append(os.path.basename(pcap))
            else:
                class_pcaps[label].append(pcap)

    if skipped:
        print(f"  [SKIP] {len(skipped)} uncertain files")

    # Extract Channel Units per class
    class_units = defaultdict(list)
    for cls, pcaps in sorted(class_pcaps.items()):
        for pcap in pcaps:
            u = pcap_to_units(
                pcap, max_units=n_cap * 2, window_l=window_l, stride_s=stride_s, mtu=mtu
            )
            class_units[cls].extend(u)
            if len(class_units[cls]) >= n_cap * 2:
                break
        print(f"  {cls}: {len(class_units[cls])} units from {len(pcaps)} pcap(s)")

    # Filter: require >= 500 units
    counts = {cls: len(units) for cls, units in class_units.items()}
    valid_classes = sorted(cls for cls, cnt in counts.items() if cnt >= 500)
    print(f"  Valid classes (≥500): {len(valid_classes)} — {valid_classes}")

    if len(valid_classes) < 5:
        print(
            f"  [WARN] Only {len(valid_classes)} valid classes (need ≥5 for 5-fold holdout)"
        )

    actual_n_cap = min(n_cap, min((counts[c] for c in valid_classes), default=0))
    print(f"  Balanced N_cap: {actual_n_cap}")

    all_units, all_labels, class_names = [], [], []
    for label_idx, cls in enumerate(valid_classes):
        units = class_units[cls]
        if len(units) > actual_n_cap:
            random.seed(seed)
            units = random.sample(units, actual_n_cap)
        else:
            units = units[:actual_n_cap]
        all_units.extend(units)
        all_labels.extend([label_idx] * len(units))
        class_names.append(cls)

    vocab_size = 2 + 2 * mtu + 1
    padded = []
    for u in all_units:
        if len(u) < window_l:
            u = u + [0] * (window_l - len(u))
        padded.append(u[:window_l])

    X = np.array(padded, dtype=np.int32)
    y = np.array(all_labels, dtype=np.int64)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "features.npy"), X)
    np.save(os.path.join(cache_dir, "labels.npy"), y)

    with open(os.path.join(cache_dir, "class_stats.json"), "w") as f:
        json.dump(
            {
                "class_names": class_names,
                "vocab_size": vocab_size,
                "class_counts": {
                    name: int(np.sum(y == i)) for i, name in enumerate(class_names)
                },
                "n_valid_classes": len(valid_classes),
                "actual_n_cap": actual_n_cap,
                "skipped_files": skipped,
            },
            f,
            indent=2,
        )

    with open(meta_path, "w") as f:
        json.dump(
            {
                "n_cap": n_cap,
                "window_l": window_l,
                "stride_s": stride_s,
                "mtu": mtu,
                "seed": seed,
                "root": root,
                "subset": subset,
                "label_scheme": label_scheme,
                "timestamp": str(np.datetime64("now")),
            },
            f,
            indent=2,
        )

    print(f"Saved to cache: {cache_dir}")
    return X, y, class_names, vocab_size


def split_data(X, y, novel_idx, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    60/20/20 split with class-holdout protocol.
    Novel class goes entirely to test set.

    Returns:
        X_train, y_train, X_val, y_val, X_test_k, y_test_k, X_novel, y_novel, n_classes
    """
    np.random.seed(seed)

    known_mask = y != novel_idx
    novel_mask = y == novel_idx

    Xk, yk = X[known_mask], y[known_mask]
    Xn, yn = X[novel_mask], y[novel_mask]

    # Remap known labels to 0..K-1
    known_classes = sorted(set(yk.tolist()))
    remap = {c: i for i, c in enumerate(known_classes)}
    yk_r = np.array([remap[c] for c in yk])

    n = len(yk_r)
    idx = np.random.permutation(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    X_train, y_train = Xk[train_idx], yk_r[train_idx]
    X_val, y_val = Xk[val_idx], yk_r[val_idx]
    X_test_k, y_test_k = Xk[test_idx], yk_r[test_idx]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test_k,
        y_test_k,
        Xn,
        yn,
        len(known_classes),
    )
