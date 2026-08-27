# SepSpace

Official PyTorch implementation of **SepSpace**, a payload-free framework for open-set network traffic classification. SepSpace represents traffic with packet-length and direction sequences, learns discriminative representations with supervised contrastive learning, and rejects unseen classes using calibrated class-specific prototype radii.

**Paper:** [SepSpace: payload-free open-set network traffic classification via supervised contrastive representation learning and prototype-radius rejection](https://doi.org/10.1186/s42400-026-00635-x), *Cybersecurity*, 2026.

## Repository scope

This repository releases the implementation and experiment configuration. It does **not** redistribute third-party PCAP files, preprocessed feature caches, trained checkpoints, or third-party baseline implementations. Download each dataset from its original provider and comply with its terms of use, citation requirements, and any access form.

## Requirements

- Python 3.8 or later
- PyTorch 2.0 or later
- An NVIDIA GPU is recommended for the full experiments, but the code falls back to CPU

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA or CPU environment if the default `pip` wheel is not suitable. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Datasets

All raw data are obtained from their original sources. The loader converts PCAP/PCAPNG files into payload-free Channel Units with a window length of 8 packets, stride 4, and MTU 1500. Generated caches are stored locally under `data/preprocessed/` and are intentionally ignored by Git.

| Dataset | Role in the paper | Paper configuration | Official source |
| --- | --- | --- | --- |
| USTC-TFC2016 | Mixed benign/malicious traffic | 20 classes, `N_cap=5000` | [USTC-TFC2016 repository](https://github.com/yungshenglu/USTC-TFC2016) |
| MCFP | Malware-family traffic plus Normal-32 | 8 classes, `N_cap=5000` | [Malware Capture Facility Project index](https://mcfp.felk.cvut.cz/publicDatasets/datasets.html) |
| ISCX VPN-nonVPN 2016 | Application traffic, split into VPN and non-VPN application-level datasets | 14 non-VPN app classes, `N_cap=1024`; 14 VPN app classes, `N_cap=611` | [CIC/ISCX VPN-nonVPN 2016](https://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/) |

### Expected raw-data layouts

The loaders expect the following directory layouts after download and extraction:

```text
USTC-TFC2016/
  Malware/*.pcap
  Benign/*.pcap

MCFP/
  Cobalt/*.pcap
  Dridex/*.pcap
  Normal-32/*.pcap
  TrickBot/*.pcap
  Trojan_Downloader/*.pcap
  Trojan_Locky/*.pcap
  Worm_Allape/*.pcap
  Zeus/*.pcap

ISCX_VPN-nonVPN_2016/
  VPN-PCAPS-01/*.{pcap,pcapng}
  VPN-PCAPs-02/*.{pcap,pcapng}
  NonVPN-PCAPs-01/*.{pcap,pcapng}
  NonVPN-PCAPs-02/*.{pcap,pcapng}
  NonVPN-PCAPs-03/*.{pcap,pcapng}
```

MCFP is a curated benchmark assembled from the public MCFP captures. The class names above identify the required locally organized subsets; use the MCFP index to retrieve the corresponding captures.

### Other paper settings

```bash
# MCFP
python run_experiment.py --dataset mcfp --data_root /path/to/MCFP \
  --novel_class Cobalt --n_cap 5000 --method sepspace --seed 42

# ISCX non-VPN application labels
python run_experiment.py --dataset iscx-nonvpn-app --data_root /path/to/ISCX_VPN-nonVPN_2016 \
  --novel_class email --n_cap 1024 --method sepspace --seed 42

# ISCX VPN application labels
python run_experiment.py --dataset iscx-vpn-app --data_root /path/to/ISCX_VPN-nonVPN_2016 \
  --novel_class aim --n_cap 611 --method sepspace --seed 42
```

The supported methods are `sepspace`, `ce_msp`, `ce_energy`, `ce_pr`, `ce_mahalanobis`, `sepspace_frozen`, `sepspace_lof`, and `nnfst`.

## Project layout

```text
models.py             SepSpace encoder, projection head, and training routines
data_loader.py        PCAP parsing, Channel Unit construction, and data splits
evaluate.py           Open-set metrics and baselines
nnfst.py              nNFST baseline adapter
run_experiment.py     Reproducible single-fold experiment entry point
requirements.txt      Runtime dependencies
```

## Citation

```bibtex
@article{qu2026sepspace,
  title={SepSpace: payload-free open-set network traffic classification via supervised contrastive representation learning and prototype-radius rejection},
  author={Qu, Yanze and Zheng, Chaofan and Liao, Minxi and Ma, Hailong and Jiang, Yiming and Wang, Wenbo},
  journal={Cybersecurity},
  volume={9},
  pages={204},
  year={2026},
  doi={10.1186/s42400-026-00635-x}
}
```
