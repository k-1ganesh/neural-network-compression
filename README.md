# Neural Network Compression with SVD, CP and Tucker Decomposition
> **Study of low-rank and tensor-based compression for an MLP**

This project investigates how much a trained neural network can be compressed by exploiting structure in its learned weight matrices.

The experiment starts with a dense Multi-Layer Perceptron (MLP) trained on **Fashion-MNIST**. Four internal weight matrices are grouped into a third-order tensor, and three different decomposition strategies are applied:

- **SVD** on a column-wise concatenation of the hidden-layer weight matrices

- **CP (CANDECOMP/PARAFAC)** decomposition of the resulting weight tensor

- **Tucker** decomposition of the same tensor

For each decomposition, the rank is varied and the compressed weights are reconstructed and inserted back into the original network. The compressed network is then evaluated **without retraining** so that the direct effect of the decomposition on model accuracy can be measured.

---

## Table of Contents
- [Motivation](#motivation)
- [Experiment Overview](#experiment-overview)
- [Dense Baseline](#dense-baseline)
- [Global Weight Tensor](#global-weight-tensor)
- [Compression Methods](#compression-methods)
  - [SVD](#1-svd)
  - [CP Decomposition](#2-cp-decomposition)
  - [Tucker Decomposition](#3-tucker-decomposition)
- [Experimental Protocol](#experimental-protocol)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Results Discussion](#results-discussion)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Experiment](#running-the-experiment)
---

## Motivation
A standard neural network stores every learned weight independently. This can result in a large number of parameters even when the learned weight matrices contain substantial redundancy or low-dimensional structure.

Low-rank matrix decomposition and tensor decomposition provide a way to represent these weights using fewer numbers.

The central question of this project is:

> **How aggressively can a trained neural network be compressed while preserving its predictive accuracy?**

The experiment is deliberately structured so that the comparison is made from the **same trained dense model**. This allows the effect of SVD, CP and Tucker compression to be studied without changing the network architecture or retraining the dense baseline separately for each method.

---

## Experiment Overview
The complete pipeline is:

```text

Fashion-MNIST

      │

      ▼

Train Dense MLP

      │

      ▼

Save Dense Checkpoint

      │

      ▼

Extract Four 512×512 Hidden Weight Matrices

      │

      ▼

Construct Global Weight Tensor

      │

      ├──────────────┬───────────────┐

      ▼              ▼               ▼

     SVD             CP            Tucker

      │              │               │

      ▼              ▼               ▼

   Rank Sweep      Rank Sweep    Rank Configuration Sweep

      │              │               │

      └──────────────┴───────────────┘

                     │

                     ▼

          Reconstruct Hidden Weights

                     │

                     ▼

          Insert into Original MLP

                     │

                     ▼

              Test Accuracy

                     │

                     ▼

       Reconstruction / Compression Analysis

```

The important controlled variable is the decomposition rank.

For SVD and CP, a single rank is varied. For Tucker, a multilinear rank configuration is varied.

---

# Dense Baseline
The baseline model is a fully connected MLP with the following architecture:

```text

784 → 512 → 512 → 512 → 512 → 512 → 10

```

The input is a flattened Fashion-MNIST image:

$$
28\times 28 = 784
$$

features.

The model uses ReLU activations between the hidden linear layers and a final linear classification layer with 10 outputs.

### Training setup
The dense network is trained once and then frozen as the reference model.

The experiment uses a fixed training configuration rather than performing hyperparameter search:

| Setting | Value |
|---|---:|
| Dataset | Fashion-MNIST |
| Architecture | 784 → 512 → 512 → 512 → 512 → 512 → 10 |
| Optimizer | AdamW |
| Learning rate | $10^{-3}$ |
| Weight decay | $10^{-4}$ |
| Batch size | 256 |
| Epochs | 20 |
| Hidden activation | ReLU |
| Loss | Cross-Entropy |

The dense checkpoint is used as the starting point for **every compression run**.

This is important: an SVD/CP/Tucker experiment is never initialized from another compressed model. Every rank is evaluated from the same original dense weights.

---

# Global Weight Tensor

We select the four internal $512 \times 512$ weight matrices:

$$
W_1, W_2, W_3, W_4 \in \mathbb{R}^{512 \times 512}.
$$

These matrices are stacked along a third mode to form:

$$
\boxed{\mathcal{W} \in \mathbb{R}^{512 \times 512 \times 4}}
$$

with

$$
\begin{aligned}
\mathcal{W}[:,:,1] &= W_1, \\
\mathcal{W}[:,:,2] &= W_2, \\
\mathcal{W}[:,:,3] &= W_3, \\
\mathcal{W}[:,:,4] &= W_4.
\end{aligned}
$$

The four matrices contain:

$$
4 \times 512 \times 512 = 1,048,576
$$

weights.

This **1,048,576-parameter block** is the part compressed in the current experiment.

The first input layer, the final classification layer, and the biases are left unchanged in this experiment.

---

# Compression Methods
## 1. SVD
The four hidden matrices are concatenated column-wise:

$$
M=
[W_1\;W_2\;W_3\;W_4]
$$

so that

$$
M\in\mathbb{R}^{512\times2048}.
$$

We compute the singular value decomposition:

$$
M=U\Sigma V^T.
$$

For a chosen rank $r$, the truncated approximation is:

$$
M_r=U_r\Sigma_rV_r^T.
$$

The reconstructed matrix is then split back into four $512\times512$ matrices:

$$
M_r=
[\hat W_1\;\hat W_2\;\hat W_3\;\hat W_4].
$$

### SVD rank sweep
The experiment evaluates multiple values of $r$:

$$
r\in
\{1,2,4,8,16,32,64,128,256,384,512\}.
$$

This makes it possible to observe how the reconstruction error and network accuracy change as more rank is retained.

---

## 2. CP Decomposition
The same tensor

$$
\mathcal{W}\in\mathbb{R}^{512\times512\times4}
$$

is approximated using CP decomposition:

$$
\mathcal{W}
\approx
\displaystyle\sum_{k=1}^{R}
a_k \circ b_k \circ c_k
$$

Here $R$ is the CP rank and $\circ$ denotes the vector outer product.

The factor matrices have dimensions:

$$
A\in\mathbb{R}^{512\times R},
$$

$$
B\in\mathbb{R}^{512\times R},
$$

$$
C\in\mathbb{R}^{4\times R}.
$$

The reconstructed tensor is converted back to the four hidden weight matrices before being inserted into the MLP.

### CP rank sweep
The experiment evaluates:

$$
R\in
\{1,2,4,8,16,32,64,128,256\}.
$$

CP is optimized iteratively, so unlike SVD, its decomposition quality depends on the numerical optimization procedure and initialization.

---

## 3. Tucker Decomposition
The same global tensor is decomposed as:

$$
\mathcal{W}
\approx
\mathcal{G}
\times_1 U_1
\times_2 U_2
\times_3 U_3.
$$

The factor matrices have dimensions:

$$
U_1\in\mathbb{R}^{512\times r_1},
$$

$$
U_2\in\mathbb{R}^{512\times r_2},
$$

$$
U_3\in\mathbb{R}^{4\times r_3},
$$

and the core tensor is:

$$
\mathcal{G}\in
\mathbb{R}^{r_1\times r_2\times r_3}.
$$

For this experiment we use:

$$
r_1=r_2
$$

and vary the third-mode rank independently.

Since the third mode represents four hidden layers:

$$
r_3\le4.
$$

The Tucker experiment therefore explores configurations such as:

$$
(4,4,1),\;(4,4,2),\;(4,4,4),
$$

$$
(16,16,1),\;(16,16,2),\;(16,16,4),
$$

$$
(32,32,1),\;(32,32,2),\;(32,32,4),
$$

and larger configurations.

---

# Experimental Protocol
For every decomposition and every rank/configuration:

1\. Start from the **same dense checkpoint**.

2\. Extract $W_1,\ldots,W_4$.

3\. Apply the selected decomposition.

4\. Reconstruct $\hat W_1,\ldots,\hat W_4$.

5\. Replace the original hidden weights with the reconstructed weights.

6\. Leave all other network parameters unchanged.

7\. Evaluate the resulting model on the Fashion-MNIST test set.

No fine-tuning is performed during this first experiment.

Therefore the measured accuracy answers:

> **How much of the original network's predictive performance survives the weight approximation alone?**

---

# Evaluation Metrics
Several metrics are recorded.

## Reconstruction Error
The relative Frobenius reconstruction error is:

$$
\boxed{
E=
\frac{\|\mathcal{W}-\hat{\mathcal{W}}\|F}
{\|\mathcal{W}\|F}
}
$$

For SVD, the analogous calculation is performed on the concatenated matrix $M$.

Lower values indicate a closer approximation to the original weights.

---

## Compressed Parameter Count
For each decomposition, we count the parameters required to store its factors/core rather than the reconstructed dense matrices.

For example, the global SVD representation has:

$$
512r + 2048r = 2560r
$$
stored factor parameters.

For CP, the factor storage is approximately:

$$
512R+512R+4R.
$$

For Tucker:

$$
512r_1+
512r_2+
4r_3+
r_1r_2r_3.
$$

---

## Compression Ratio
For the reported table, the compression ratio is computed **only for the compressible hidden-weight block**:

$$
\boxed{
\text{Compression Ratio}
=
\frac{1,048,576}
{\text{compressed block parameters}}
}
$$

This is not the compression ratio of the entire neural network, because the remaining layers and biases are intentionally left dense.

---

## Test Accuracy
The final metric is the Fashion-MNIST test accuracy after replacing the original hidden weights with their decompressed approximations.

---

# Results
The following table reports representative high-compression configurations from the completed experiment.

| Method | Compressible Weight Block | Compressed Weight Block | Compression Ratio | Test Accuracy |
|---|---:|---:|---:|---:|
| **Dense (Baseline)** | 1,048,576 | 1,048,576 | **1.00×** | **89.06%** |
| **SVD (r=64)** | 1,048,576 | 163,840 | **6.40×** | **88.72%** |
| **CP (R=128)** | 1,048,576 | 131,584 | **7.97×** | **89.24%** |
| **Tucker (64,64,4)** | 1,048,576 | 81,936 | **12.80×** | **88.76%** |

### Key observations
At the selected operating points:

- **SVD (r=64)** reduces the selected hidden-weight block by **6.40×** while retaining **88.72%** test accuracy, only **0.34 percentage points** below the dense baseline.

- **CP (R=128)** reduces the block by **7.97×** and obtains **89.24%** test accuracy, which is **0.18 percentage points above** the reported dense baseline on this run.

- **Tucker (64,64,4)** achieves the largest compression in this table, at **12.80×**, while retaining **88.76%** accuracy, a **0.30 percentage-point** decrease from the dense baseline.

The results show that a substantial amount of redundancy exists in the learned hidden-layer weights: the network can tolerate significant approximation of these matrices while keeping test accuracy close to the original model.

> **Important:** the table compares compression of the selected $1,048,576$-parameter hidden-weight block. It does not claim that the entire MLP has been compressed by the same ratio.

---

# Rank and Compression Curves
The experiment also evaluates a range of ranks rather than selecting a single rank in advance.

The main analyses are:

### Rank vs Test Accuracy
This shows the accuracy transition as more decomposition rank is retained.

### Rank vs Reconstruction Error
This shows how rapidly each decomposition improves its approximation as rank increases.

### Compression Ratio vs Test Accuracy
This is the primary cross-method comparison because SVD rank, CP rank and Tucker multilinear rank are not directly comparable. Compression ratio puts the methods on a common storage-based scale.

### Compression Ratio vs Reconstruction Error
This shows how much approximation error is incurred for a given reduction in stored parameters.

The generated plots are stored in the \`plots/\` directory.

---

# Repository Structure
```text

nn_tensor_compression/

│

├── data.py

├── model.py

├── train.py

├── evaluate.py

├── utils.py

│

├── compression/

│   ├── __init__.py

│   ├── svd.py

│   ├── cp.py

│   └── tucker.py

│

├── experiments/

│   ├── __init__.py

│   └── run_compression.py

│

├── plots/

│   └── plot_results.py

│

├── checkpoints/

│   └── dense_final.pt

│

├── results/

│   └── compression_results.csv

│

└── requirements.txt

```

# Installation
## Requirements
The experiment was designed to run on a consumer GPU and does not require a large compute cluster for this network size.

Install the dependencies with:

```bash

pip install -r requirements.txt

```

The main packages are:

```text

torch

torchvision

tensorly

numpy

pandas

matplotlib

tqdm

```

---

# Running the Experiment
## 1. Train the dense baseline
```bash

python train.py

```

This creates the dense model checkpoint:

```text

checkpoints/dense_final.pt

```

---

## 2. Run SVD, CP and Tucker sweeps
From the project root:

```bash

python -m experiments.run_compression

```

The experiment evaluates all configured ranks and Tucker rank combinations.

The complete results are written to:

```text

results/compression_results.csv

```

---

## 3. Generate plots
```bash

python plots/plot_results.py

```

The plots are written to:

```text

plots/

```

including:

```text

rank_vs_accuracy.png

rank_vs_reconstruction_error.png

compression_vs_accuracy.png

compression_vs_reconstruction_error.png

```

---
