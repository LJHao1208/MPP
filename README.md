# MPP: A Multi-Scale Pattern Probe for Efficient Respiratory Sound Measurement

This repository contains the official PyTorch implementation of **MPP (Multi-scale Pattern Probe)**, a lightweight probing framework for efficient respiratory sound measurement. MPP keeps a pre-trained Audio Spectrogram Transformer (AST) encoder fully frozen and trains only a 2.38M-parameter probe, achieving competitive performance with significantly fewer trainable parameters than full fine-tuning.

## 📋 Table of Contents

- [Introduction](#introduction)
- [Method Overview](#method-overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

Respiratory sound classification is essential for early screening of pulmonary diseases, but clinical datasets are often small, noisy, and severely imbalanced. Existing methods predominantly rely on full-parameter fine-tuning of large pre-trained models, which is computationally expensive and prone to overfitting. MPP addresses this by keeping a pre-trained AST encoder fully frozen and training only a lightweight probe, reducing trainable parameters to **2.38M** (37× fewer than full fine-tuning) while achieving a **62.28** Score on ICBHI 2017 and **61.75** on SPRSound under zero-shot transfer.

## Method Overview

MPP consists of two core modules:

1. **Multi-Scale Token Aggregator (MTA)**: Extracts token sequences from intermediate layers 3, 6, 9, and 12 of the frozen AST encoder, applies attention pooling with layer-specific learnable queries, and combines them with learnable scale weights. This preserves both fine-grained transient events (e.g., crackles) and high-level semantic information.

2. **Pattern Proto Probe (PPP)**: Learns K=4 prototypes per class and classifies via maximum cosine similarity. Compared with a linear classification head, the prototype probe provides more flexible decision boundaries under severe class imbalance.

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/LJHao1208/MPP.git
cd MPP

# Create conda environment
conda create -n mpp python=3.8
conda activate mpp

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib
pip install librosa soundfile

## Data Preparation

### ICBHI 2017 Dataset

Download the ICBHI 2017 Respiratory Sound Database from the official challenge page:

```bash
# Download ICBHI 2017 dataset
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip --no-check-certificate
unzip ICBHI_final_database.zip -d ./data/ICBHI
```

The dataset should be organized as:

```
data/ICBHI/
├── ICBHI_final_database/
│   ├── *.wav
│   └── *.txt
├── metadata.txt
└── official_split.txt
```

### SPRSound Dataset

Download the SPRSound dataset from the official repository:

```bash
git clone https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git
```

Organize the data as:

```
data/SPRSound-main/
├── BioCAS2022/
├── BioCAS2023/
├── BioCAS2024/
├── BioCAS2025/
└── Classification/
```

## Training

### Main Experiments (ICBHI)

To train MPP on ICBHI 2017 with the official 60/40 split:

```bash
python main.py \
  --dataset icbhi \
  --seed 1 \
  --class_split lungsound \
  --n_cls 4 \
  --epochs 50 \
  --batch_size 8 \
  --optimizer adam \
  --learning_rate 1e-4 \
  --weight_decay 1e-6 \
  --cosine \
  --model ast \
  --test_fold official \
  --pad_types repeat \
  --resz 1 \
  --n_mels 128 \
  --from_sl_official \
  --audioset_pretrained \
  --method hp3 \
  --n_prototypes 4 \
  --agg_layers 3,6,9,12 \
  --lambda_sep 0.1 \
  --lambda_div 0.01
```

### Ablation Studies

For ablation studies on layer selection, prototype count, loss terms, and classification head:

```bash
# Multi-scale layers
python main.py ... --agg_layers 12          # single layer
python main.py ... --agg_layers 6,12        # two layers

# Prototype count
python main.py ... --n_prototypes 1
python main.py ... --n_prototypes 8

# Loss terms
python main.py ... --lambda_div 0           # without diversity loss
python main.py ... --lambda_sep 0           # without separation loss
python main.py ... --exclude_special_tokens # without special tokens

# Classification head
python main.py ... --hp3_use_linear_head    # linear head
```

## Evaluation

### Cross-Dataset Transfer (ICBHI → SPRSound)

To evaluate zero-shot transfer from ICBHI to SPRSound:

```bash
python analysis/cross_dataset_eval.py \
  --ckpt ./save/icbhi_ast_hp3_bs8_lr1e-4_ep50_seed1/best.pth \
  --sprsound_root ./data/SPRSound-main \
  --save_folder ./save/sprsound_eval_seed1 \
  --n_cls 4 \
  --n_prototypes 4 \
  --agg_layers 3,6,9,12 \
  --batch_size 16 \
  --num_workers 8
```

### Cross-Device Evaluation

To evaluate per-device performance on ICBHI:

```bash
python main.py ... --report_per_device --eval --pretrained \
  --pretrained_ckpt ./save/icbhi_ast_hp3_bs8_lr1e-4_ep50_seed1/best.pth
```

## Project Structure

```
MPP/
├── main.py                          # Main training/evaluation entry
├── models/
│   └── ast.py                       # AST encoder with return_hidden support
├── method/
│   └── hp3.py                       # MPP core modules (MTA + PPP)
├── util/
│   ├── icbhi_dataset.py             # ICBHI data loader
│   ├── icbhi_util.py                # Evaluation functions (get_score)
│   ├── sprsound_dataset.py          # SPRSound data loader
│   ├── sprsound_util.py             # SPRSound utilities
│   └── analysis.py                  # Cross-device collector
├── analysis/
│   ├── cross_dataset_eval.py        # Cross-dataset evaluation
│   ├── plot_confusion.py            # Confusion matrix visualization
│   ├── plot_logits.py               # Logit distribution plots
│   ├── visualize_proto.py           # t-SNE prototype visualization
│   └── stats_test.py                # Statistical significance tests
├── scripts/
│   ├── main/
│   │   └── mpp.sh                   # Main experiment script
│   ├── ablation/
│   │   ├── layers.sh                # Layer selection ablation
│   │   ├── k.sh                     # Prototype count ablation
│   │   ├── loss.sh                  # Loss terms ablation
│   │   └── head.sh                  # Classification head ablation
│   ├── cross_device/
│   │   └── per_device.sh            # Cross-device evaluation
│   └── cross_dataset/
│       └── sprsound.sh              # Cross-dataset evaluation
├── data/                            # Dataset directory
├── save/                            # Experiment outputs
└── figs/                            # Figures for paper
```

## Results

### Main Results on ICBHI 2017 (Official 60/40 Split)

| Method | Params | S_p (%) | S_e (%) | Score (%) |
|--------|--------|---------|---------|-----------|
| PatchMix-CL | 88.71M | 81.76±5.16 | 41.35±4.63 | 61.55±0.44 |
| LungAdapter | 2.45M | 79.63±0.83 | 42.14±0.64 | 60.89±0.32 |
| LoRA-RSC | 0.30M | 78.93±2.51 | 41.92±2.49 | 60.42±0.10 |
| AST-FT | 87.53M | 75.64±2.99 | 44.69±2.73 | 60.16±1.33 |
| PatchMix-CE | 87.53M | 78.28±5.08 | 40.78±3.92 | 59.53±0.95 |
| Frozen-CE | 4.6K | 52.16±2.93 | 51.12±1.50 | 51.64±0.86 |
| **MPP (Ours)** | **2.38M** | **82.46±2.36** | **42.11±2.61** | **62.28±0.12** |

### Cross-Dataset Transfer (ICBHI → SPRSound)

| Method | Params | S_p (%) | S_e (%) | Score (%) |
|--------|--------|---------|---------|-----------|
| BTS-CARD | 158M | 82.02 | 41.90 | 61.96±1.50 |
| **MPP (Ours)** | **2.38M** | **88.46** | **35.04** | **61.75±0.59** |
| QLung (Audio-CLAP) | 28M | 74.71 | 44.88 | 59.80±3.51 |
| Audio-CLAP | 28M | 70.67 | 41.90 | 56.29 |
| BTS | 158M | 67.50 | 39.33 | 53.42 |
| LungAdapter | 2.45M | 55.38±7.21 | 39.85±7.85 | 47.62±6.27 |

### Ablation Studies

| Configuration | S_p (%) | S_e (%) | Score (%) |
|---------------|---------|---------|-----------|
| Single layer (12) | 77.96±1.79 | 41.30±1.92 | 59.63±0.44 |
| Two layers (6,12) | 80.92±2.23 | 40.74±1.79 | 60.83±0.65 |
| Four layers (3,6,9,12) | 82.46±2.36 | 42.11±2.61 | 62.28±0.12 |
| K=1 | 79.71±2.35 | 42.43±2.49 | 61.07±0.14 |
| K=4 | 82.46±2.36 | 42.11±2.61 | 62.28±0.12 |
| K=8 | 76.29±5.56 | 44.95±3.94 | 60.62±1.32 |
| Linear head | 74.58±5.06 | 43.98±2.52 | 59.28±1.28 |
| Proto probe (Ours) | 82.46±2.36 | 42.11±2.61 | 62.28±0.12 |

## Acknowledgments

The authors would like to thank the providers of the ICBHI 2017 and SPRSound datasets for making the data publicly available. We also thank the authors of PatchMix for releasing their codebase, which facilitated part of our experiments.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Jiahao Li: 241020070@fzu.edu.cn
- Shu Zhang: zhangshu@fzu.edu.cn
```
