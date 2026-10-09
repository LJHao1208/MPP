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
