# Hybrid Quantum-Classical Pneumonia Detection
AI505 — Assignment 3 | Group Project

---

## Description

This project implements a hybrid quantum-classical transfer learning pipeline for binary chest X-ray classification (Pneumonia vs. Normal). The classical backbone ResNet-18 is used as a frozen feature extractor, and its final fully connected layer is replaced by a 4-qubit Variational Quantum Circuit (VQC) built with PennyLane. The goal is to evaluate whether a minimal quantum layer can match or approach the diagnostic performance of a fully classical baseline under identical data and hardware conditions.

This work constitutes Assignment 3 of AI505, building directly on the critical literature survey conducted in Assignment 2, which identified hybrid transfer learning as the most viable near-term approach for Quantum Machine Learning in medical imaging.

---

## Architecture

**Classical Baseline:** ResNet-18 (frozen) + Linear(512, 1) + Sigmoid — trained with Adam and Binary Cross-Entropy.

**Hybrid Quantum Model:** ResNet-18 (frozen) + Linear(512, 4) + Tanh scaling + 4-qubit VQC (AngleEmbedding, BasicEntanglerLayers, PauliZ measurement) via PennyLane default.qubit + Sigmoid output.

The forward pass bridges CUDA and CPU tensors to accommodate PennyLane's CPU-only default.qubit simulator on Kaggle GPU T4.

---

## Dataset

**Name:** Chest X-Ray Images (Pneumonia)
**Source:** Kermany et al. (2018) — Cell journal.
**Kaggle Link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
**License:** CC BY 4.0
**Size:** 5,856 chest radiographs — binary labels: NORMAL / PNEUMONIA.

### Split Strategy

| Split      | NORMAL | PNEUMONIA | Total |
|------------|--------|-----------|-------|
| Training   | 600    | 600       | 1,200 |
| Validation | 120    | 120       | 240   |
| Test       | 234    | 390       | 624   |

Balanced subsampling is applied to accommodate quantum simulation overhead and ensure full execution within 30 minutes on Kaggle GPU T4.

---

## Results

All models evaluated on the same held-out test set (624 images). Random seed fixed at 42 for full reproducibility.

| Diagnostic Metric    | Classical Baseline | Hybrid Quantum |
|----------------------|--------------------|----------------|
| Accuracy             | 0.8446             | 0.8381         |
| AUC-ROC              | 0.9467             | 0.9199         |
| F1-Score             | 0.8849             | 0.8802         |
| Precision            | 0.8234             | 0.8190         |
| Sensitivity (Recall) | 0.9564             | 0.9513         |
| Specificity          | 0.6581             | 0.6496         |
| Inference Time (ms)  | 23.28              | 22.31          |

The hybrid model achieves within 0.65 percentage points of the classical baseline on accuracy despite replacing 512 classical neurons with 4 qubits. Both models exceed 95% Sensitivity. The AUC gap of ~2.7 points is consistent with NISQ-era constraints.

---

## How to Run

1. Open the notebook on Kaggle.
2. Add the dataset: chest-xray-pneumonia (from Kaggle Datasets).
3. Enable GPU acceleration: Settings > Accelerator > GPU T4.
4. Run All Cells.

Estimated runtime: 20–28 minutes on GPU T4 with Early Stopping active.

---

## Requirements

Pre-installed on Kaggle:

```
torch, torchvision, pandas, scikit-learn, matplotlib, seaborn, numpy
```

Installed automatically in Cell 1:

```
pennylane, pennylane-lightning
```

---

## References

- Kermany et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell, 172*(5), 1122–1131.e9.
- Chan et al. (2024). Hybrid quantum-classical neural networks for medical image classification under data scarcity. *Quantum Machine Intelligence, 6*(1), 18.
- Xiang et al. (2024). Transfer learning in hybrid quantum-classical architectures. *IEEE Transactions on Neural Networks and Learning Systems.*
- PennyLane Development Team. (2024). https://pennylane.ai
