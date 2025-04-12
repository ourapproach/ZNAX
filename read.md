# Replication package for paper "Robust Device Authentication with Zero-Cost NAS and Cross-Domain Knowledge Transfer"
## 🔍 Framework Overview

![ZNAX Framework](./ZNAX.PNG)
> Figure: A Conceptual Overview 

The proposed framework addresses the neural network design complexity and data scarcity challenges of **deep learning-based Specific Emitter Identification (SEI)**. It is dubbed **ZNAX**, where **Z** stands for **Zero-Cost**, **NA** for **NAS**, and **X** for **Cross-domain**. The framework leverages **zero-cost NAS** for optimal architecture discovery and employs a **cross-domain knowledge transfer** approach to mitigate data scarcity. The **ZNAX** framework operates in three sequential stages, as shown in **the figure**.

This repository provides the implementation of **ZNAX**, a framework designed for robust and efficient Specific Emitter Identification (SEI) using:

- ⚙️ **Zero-Cost Neural Architecture Search (NAS)** to find optimal Vision Transformer (ViT) architectures without full training
- 🌐 **Cross-Domain Knowledge Transfer** to overcome RF data scarcity using contrastive learning and domain alignment

ZNAX enables lightweight, generalizable RF device authentication using minimal labeled data, making it ideal for edge computing scenarios.

---



---

## 🧠 Pipeline Stages

### 1. Zero-Cost NAS
- Searches ViT configurations based on discriminability, trainability, cohesiveness, and attention diversity
- Uses a single forward-backward pass per architecture and evolutionary ranking

### 2. Cross-Domain Pre-Training
- Transfers knowledge from a source image dataset (e.g., CIFAR-10, MNIST) to RF domain
- Uses dual-margin contrastive loss and Maximum Mean Discrepancy (MMD) to align feature distributions

### 3. Supervised Fine-Tuning
- Trains the ViT encoder and classifier on a small labeled RF dataset for final emitter classification

---

## 📦 Dependencies

Install the required packages:

```bash
pip install torch torchvision
pip install numpy scipy scikit-learn matplotlib tqdm
pip install pyyaml seaborn
