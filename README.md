# GUARDNET 🧠

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **From Global Search to Local Lookup: Scalable Knowledge Base Completion via Differentiable Guarded Logic**  
> *Under review at SIGMOD 2027*

This repository provides the official PyTorch implementation for **GUARDNET**. The code is designed to be a complete and faithful reproduction of the paper's methodology, enabling researchers to explore and build upon our work in scalable neuro-symbolic reasoning.

---

## 📖 Extended Version & Supplementary Material

> **📌 Note for Reviewers:** This repository contains the **extended version** of the paper with additional materials not included in the main submission due to page limits:
> - **Appendix A**: Theoretical Proofs and Analysis
>   - Summary of Notation
>   - Formal Semantics of GF
>   - Full Fuzzy Operator Comparison
>   - Formal Definitions of Fuzzy Operators
>   - Proof of Zero-Loss Semantic Consistency (Theorem 1)
>   - Proof of Approximate Satisfaction (Corollary 1)
>   - Proof of Training Complexity (Theorem 2)
> - **Appendix B**: Detailed Experimental Setup (dataset statistics, hyperparameters, core domain growth analysis with $|\Delta_G|$ growth curves)
> - **Appendix C**: Logic Translation Standards
> - **Appendix D**: Parameter Sensitivity Analysis
> - **Appendix E**: Algorithm Pseudocode
>
> Please refer to and download [`extended_paper.zip`](extended_paper.zip) for the complete technical details.

---

## 📋 Overview

**GUARDNET** is the first framework to leverage the **Guarded Fragment (GF)** of first-order logic as a principled inductive bias for robust and scalable neighborhood-based reasoning. It directly addresses the critical scalability bottleneck in neural-symbolic (NeSy) systems, reducing grounding complexity from $O(N^2)$ to $O(|\mathcal{E}|)$ and enabling rigorous FOL-based reasoning on knowledge bases with hundreds of thousands of entities.

The key insight is that the GF guard condition is precisely the syntactic constraint underlying **guarded tuple-generating dependencies (GTGDs)** from database theory. GUARDNET neuralizes this tractability mechanism, establishing a formal bridge between database-theoretic query answering and differentiable neural reasoning.

### 🎯 Key Features

- **Guarded Logic as Inductive Bias**: Employs the syntactic guard of GF to restrict logical quantification to local, relational neighborhoods, transforming intractable global grounding into efficient index-only lookups. This connects NeSy grounding to the DB-theoretic tractability of GTGDs.
- **Principled Fuzzy Semantics**: Built on differentiable fuzzy logic using **Product t-norms** and **Sigmoidal Reichenbach implication** with smooth, non-vanishing gradients. A new Proposition (Soundness of Masked Guarded Evaluation) and Corollary (Approximate Satisfaction) provide formal guarantees.
- **Hybrid Domain Strategy**: Novel training strategy combining **Core Domain** (Herbrand + Atom Witness + Guarded Witness) with **Latent Domain** for logical fidelity and robust generalization. Prevents representational collapse (vacuous satisfaction).
- **75× Scalability Improvement**: Scales linearly to 377K concepts (SNOMED CT) where existing NeSy methods fail at 5K entities.

---

## 🛠️ Installation & Setup

### Prerequisites

The framework is built entirely in Python. All experiments were conducted on servers equipped with NVIDIA GPUs.

| Component | Version | Purpose |
|:----------|:--------|:--------|
| **Hardware** | NVIDIA RTX 4090 (24GB) | GPU acceleration for training |
| **Python** | 3.8.0+ | Core framework language |
| **PyTorch** | 2.0+ | Deep learning backend |
| **CUDA** | 11.8+ | GPU support |

### Dependencies
```bash
# Clone the repository
git clone https://anonymous.4open.science/r/sigmod2026-271/
cd guardnet

# Install Python dependencies
pip install -r requirements.txt
```

<details>
<summary>📦 Full Dependency List</summary>

<pre>
torch>=2.0.0
numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
tqdm>=4.65.0
loguru>=0.7.0
pyparsing>=3.0.9
</pre>

</details>

---

## 🔧 Data Preparation

The framework expects standard knowledge base completion datasets, split into training, validation, and test sets. 

### 📊 Benchmark Datasets

| Dataset | Type | Scale | Description |
|:--------|:-----|:------|:------------|
| **SNOMED CT** | TBox | 377K concepts | Medical terminology ontology |
| **Gene Ontology (GO)** | TBox | 44K concepts | Biological taxonomy |
| **Yeast PPI** | ABox | 6.4K entities, 110K edges | Protein-protein interactions |
| **Human PPI** | ABox | 17.6K entities, 75K edges | Protein-protein interactions |

All datasets are randomly split into 80% training, 10% validation, and 10% test sets.

### 📁 Required Data Structure

Please place your datasets inside a `data/` directory, following this structure for each dataset:
```
data/your_dataset_name/
├── entities.txt      # List of all entity/concept names, one per line
├── relations.txt     # List of all relation (predicate) names
├── axioms.txt        # TBox axioms in GF format (e.g., ∀x(Pneumonia(x)→∃y(hasFinding(x,y)∧Fever(y))))
├── train.txt         # Training facts (head_entity relation tail_entity)
├── valid.txt         # Validation facts
└── test.txt          # Test facts
```

---

## 🚀 Training & Evaluation

The code has been structured into separate files for clarity. The main entry point is `main.py`.

### 📊 Knowledge Base Completion (KBC)

This is the primary task for evaluating standard performance on TBox reasoning (concept subsumption) and ABox reasoning (link prediction).

#### 1. Training

Use `main.py` to launch training. The script will load the configuration from `config.py`, initialize the `GuardNet` model, and run the training loop defined in `train.py`. 

The best model checkpoint is saved based on validation MRR with early stopping (patience of 15 epochs).
```bash
# Example: Training GUARDNET on SNOMED CT
python main.py --dataset snomed_ct --batch_size 512

# Example: Training GUARDNET on Yeast PPI
python main.py --dataset yeast_ppi --batch_size 1024
```

<details>
<summary>🔧 Advanced Training Configuration</summary>

The `config.py` file centralizes all hyperparameters from the paper. You can modify it directly to tune the model.

Key parameters include:

- `embedding_dim`: Dimensionality of entity embeddings (default: 200)
- `learning_rate`: Initial learning rate for AdamW (default: 2e-4)
- `batch_size`: Number of instances per batch (512 for SNOMED CT, 1024 for others)
- `weight_decay`: L2 regularization coefficient (default: 1e-5)
- `lambda_fidelity`: Weight for fidelity loss (default: 0.7)
- `gamma_conf`: Weight for confidence penalty (default: 0.01)
- `sigmoid_steepness`: Steepness parameter s for Sigmoidal implication (default: 10)
- `sigmoid_bias`: Bias parameter b₀ for Sigmoidal implication (default: -0.5)
- `p_aggregation`: Exponent p for generalized mean quantifiers (default: 2)
- `neg_samples`: Number of negative samples per positive instance (default: 64)
- `margin`: Margin δ for self-adversarial loss (default: 1.0)

</details>

#### 2. Evaluation

The evaluation logic is integrated into the training loop in `train.py`, which reports metrics (MRR, Hits@1/10/100) after each epoch.

For standalone evaluation, you can load a saved checkpoint and run it on the test set:
```bash
python evaluate.py \
    --model_path checkpoints/best_model.pt \
    --data_path data/your_dataset_name
```

The evaluation follows the standard **filtered ranking protocol**, removing known true facts from candidate lists during ranking.

---

## ⚙️ Hyperparameter Configuration

The optimal hyperparameters found in the paper are set as defaults in the `config.py` file.

| Parameter | Value | Description |
|:----------|:------|:------------|
| **Optimizer** | AdamW | Decoupled weight decay optimizer |
| **Learning Rate** | 2×10⁻⁴ | Initial learning rate |
| **Embedding Dim** | 200 | Dimensionality of entity embeddings |
| **Batch Size** | 512 / 1024 | 512 for SNOMED CT, 1024 for others |
| **L2 Regularization** | 10⁻⁵ | Weight decay coefficient |
| **Predicate MLP** | 2 layers, 256 units, ReLU | Architecture for grounding predicates |
| **T-norm** | Product | Fuzzy conjunction operator |
| **Implication** | Sigmoidal Reichenbach | s=10, b₀=-0.5 |
| **Quantifier Aggregation** | Generalized Mean (p=2) | For both ∀ and ∃ |
| **Fidelity Weight (λ)** | 0.7 | Balance between fidelity and generalization |
| **Confidence Penalty (γ)** | 0.01 | Encourages decisive predictions |
| **Negative Samples (ω)** | 64 | Corrupted samples per positive instance |
| **Margin (δ)** | 1.0 | Self-adversarial negative sampling margin |
| **Early Stopping** | 15 epochs | Patience based on validation MRR |

---

## 📈 Main Results

### TBox Reasoning (Concept Subsumption)

| Model | SNOMED CT (377K) ||| GO (44K) |||
|:------|:---:|:---:|:---:|:---:|:---:|:---:|
| | H@1 | H@10 | MRR | H@1 | H@10 | MRR |
| **GUARDNET (Ours)** | **5.1** | **28.3** | **0.125** | **5.5** | **29.8** | **0.133** |
| TransBox | 3.6 | 26.8 | 0.119 | 4.1 | 27.8 | 0.126 |
| Box2EL | 3.4 | 25.5 | 0.114 | 3.9 | 26.5 | 0.120 |
| ELEM | 2.1 | 20.1 | 0.078 | 2.4 | 23.8 | 0.089 |
| NBFNet | 1.2 | 10.5 | 0.055 | 1.8 | 13.0 | 0.071 |
| LTN | DNF | DNF | DNF | DNF | DNF | DNF |

### ABox Reasoning (Link Prediction)

| Model | Yeast PPI (110K edges) ||| Human PPI (75K edges) |||
|:------|:---:|:---:|:---:|:---:|:---:|:---:|
| | H@1 | H@10 | MRR | H@1 | H@10 | MRR |
| **GUARDNET (Ours)** | **30.5** | **60.2** | **0.405** | **29.2** | **57.9** | **0.388** |
| TransBox | 28.5 | 57.5 | 0.385 | 26.8 | 54.8 | 0.365 |
| Box2EL | 27.1 | 55.1 | 0.368 | 25.5 | 52.3 | 0.346 |
| NBFNet | 26.5 | 53.5 | 0.351 | 24.5 | 49.1 | 0.332 |
| LTN | 17.5 | 42.5 | 0.285 | 16.0 | 39.8 | 0.265 |
| RotatE | 18.5 | 42.6 | 0.268 | 16.0 | 38.3 | 0.245 |

*DNF = Did Not Finish (OOM/Timeout after 72h)*

---

## 🔬 Scalability

GUARDNET achieves **75× improvement in scalability** over existing NeSy methods:

| Model | Max Entities (before OOM) | Memory Complexity |
|:------|:-------------------------:|:-----------------:|
| LTN | ~5K | O(N²) |
| logLTN | ~5K | O(N²) |
| Neural LP | ~10K | O(N²) |
| **GUARDNET** | **377K+** | **O(\|E\|)** |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


<div align="center">
  <sub>Built with ❤️ for the Neural-Symbolic AI Community</sub>
</div>
