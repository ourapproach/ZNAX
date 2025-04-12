# Replication package for paper "Robust Device Authentication with Zero-Cost NAS and Cross-Domain Knowledge Transfer"

![ZNAX Framework](./ZNAX.PNG)
> Figure: A Conceptual Overview 

This repository provides the code and resources for the research paper **"Robust Device Authentication with Zero-Cost NAS and Cross-Domain Knowledge Transfer."** The proposed framework leverages zero-cost Neural Architecture Search (NAS) and cross-domain knowledge transfer to address key challenges in deep learning-based Specific Emitter Identification (SEI).

---

## Framework Overview  
The proposed framework addresses the neural network design complexity and data scarcity challenges of **deep learning-based Specific Emitter Identification (SEI)**. SEI authenticate wireless devices based on their unique Radio Frequency Fingerprints (RFFs). The framework is dubbed **ZNAX**, where **Z** stands for **Zero-Cost**, **NA** for **NAS**, and **X** for **Cross-domain**. **ZNAX** leverages **zero-cost NAS** for optimal architecture discovery for RFF extraction and employs a **cross-domain knowledge transfer** approach to mitigate data scarcity. The **ZNAX** framework operates in three sequential stages, as shown in the figure. **ZNAX** facilitates lightweight architecture search and generalizable wireless device authentication with minimal labeled data, making it well-suited for edge computing scenarios.

## Pipeline Stages

### 1. Zero-Cost NAS
- Searches for the optimal ViT encoder block configuration for RFF extraction based on four designed zero-cost proxies: **Discriminability**, **Trainability**, **Cohesiveness**, and **Diversity**.
- Computes proxy scores through a single forward-backward pass, applies **non-linear aggregation** to obtain an accumulative score for ranking the sampled architectures, and employs **evolutionary search** to refine the selection—identifying the top-ranked architecture that consistently performs well across all proxies.

### 2. Pre-Training Phase
- Source dataset from a data-rich domain and target dataset from a data-scarce domain are fed to identified network to extract respective embeddings.
- Contrastive training and domain alignment are performed jointly. Specifically, the **dual-margin contrastive loss** structures the embedding space for improved discriminability, while the **Maximum Mean Discrepancy (MMD)** objective facilitates alignment between the source domain and the RF data-scarce target domain.

### 3. Supervised Fine-Tuning
- To fully adapt the RFF extractor for SEI, supervised fine-tuning is performed alongside a classifier.
- As a preprocessing step, raw RF samples are first converted into **spectrograms**, and each spectrogram is divided into patches and passed through patch embedding and positional encoding.
- The **resulting representations** are then fed into the **pre-trained network** to extract **RFFs**.
- An **MLP classifier** is placed on top of the **Transformer encoder** to map the extracted RFFs to their corresponding device labels. As fine-tuning progresses, the model gradually adapts to the target domain and enables effective **device authentication under data scarcity**.

---

## 📦 Dependencies

Install the required packages:

```bash
pip install torch torchvision
pip install numpy scipy scikit-learn matplotlib tqdm
pip install pyyaml seaborn
