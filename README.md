# QRA-Attention: Quantum-Ready Attention via RFF Kernels

This repository implements a **Quantum-Ready Attention (QRA)** mechanism using Random Fourier Features (RFF) as a kernelized approximation of RBF similarity. This approach is designed to be a near-term "quantum utility" drop-in for standard dot-product attention in Transformer architectures.

## Objective

The goal is to demonstrate that replacing standard dot-product similarity (the "ruler") with an RFF kernel similarity in the final layers of a DistilBERT model yields measurable gains in:
- **Accuracy / F1** (Primary)
- **Memory & Speed Efficiency** (Secondary)
- **Attention Coherence / Robustness** (Supporting Evidence)

This project aligns with the "quantum utility" framing: achieving near-term classical gains while providing a clear path to swap RFF feature maps with quantum feature maps in the future.

## Key Features

- **Kernelized Similarity**: Replaces $S_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}}$ with $S_{ij} = \phi(q_i)^\top \phi(k_j)$, where $\phi(x)$ is an RFF approximation.
- **Minimal Intervention**: Only patches the last two layers (L4 and L5) of DistilBERT to isolate the effect on semantic composition.
- **Quantum-Ready Design**: Uses Hilbert-space overlap via kernel feature maps, compatible with future quantum hardware integration.
- **Robustness Testing**: Evaluates performance under word dropout and synonym swaps.

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `torch` (>=2.1.0)
- `transformers` (>=4.35.0)
- `datasets` (>=2.14.0)
- `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

## Methodology

We replace the standard attention similarity with an RFF approximation of the RBF kernel:

$$k(x, y) \approx \phi(x)^\top \phi(y)$$
$$\phi(x) = \sqrt{\frac{2}{m}} \cos(Wx + b)$$

Where:
- $m$: Number of random features (compute knob).
- $\sigma$: Kernel bandwidth (geometry knob).
- $W \sim \mathcal{N}(0, \sigma^{-2})$, $b \sim \text{Uniform}(0, 2\pi)$.

## Repository Structure

```text
qra_attention/
├── qra_attention/
│   ├── kernels/
│   │   ├── rff.py            # RFF implementation
│   │   └── quantum_stub.py   # Future quantum hooks
│   ├── attention/
│   │   └── kernel_self_attention.py # Kernelized similarity logic
│   └── patching/
│       └── patch_distilbert.py      # Layer-specific intervention
├── experiments/
│   ├── imdb_train.py         # Main training script (IMDb)
│   ├── imdb_eval.py          # Robustness & metric evaluation
│   └── metrics_attention.py  # Entropy & distance bias metrics
├── notebooks/
│   └── attention_maps.ipynb  # Visualization of attention maps
├── results/                  # Tables and plots
├── README.md
├── LICENSE
└── requirements.txt
```

## Running Experiments

To train the model on IMDb with kernel attention:
```bash
python experiments/imdb_train.py --m 128 --sigma 1.0 --layers 4,5
```

### Evaluation & Metrics
The project reports:
- **Accuracy & F1 Score**
- **Peak GPU Memory Usage**
- **Attention Entropy**: $\text{H}_i = -\sum_j A_{ij} \log(A_{ij})$
- **Attention Distance Bias**: Distribution of attention mass over token distances.

## License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Reference: 2505.23860v3, 2501.15630v2*
