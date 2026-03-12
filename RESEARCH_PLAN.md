# Research Plan: Graph MetaNetwork Learned Optimizers

## Literature Analysis & State of the Art

### Key Themes from Collected Literature (35 Papers)

The collected papers span the evolution of learned optimizers from 2001 to 2025:

**Foundational Work:**
- Hochreiter et al. (2001) — early LSTM-based learning to learn via gradient descent
- Andrychowicz et al. (2016) — "Learning to learn by gradient descent by gradient descent" — established the modern meta-learning approach to optimization using RNNs
- Li & Malik (2016, 2017) — RL-based approaches to learning optimization algorithms

**Architecture & Stability:**
- Metz et al. (2018, 2019, 2020) — understanding pathologies in training learned optimizers; key work on tasks, stability, architecture, and compute
- Harrison et al. (2022) — closer look at stability, robustness, and inductive biases via dynamical systems analysis
- Amos (2022) — tutorial on amortized optimization

**Scaling & Generalization:**
- VeLO (Metz et al., 2022) — massive-scale learned optimizer (4000 TPU-months), strong meta-generalization but impractical compute requirements
- Celo (Moudgil et al., 2025) — state-of-the-art meta-generalization with only 24 GPU hours; identifies key architecture elements
- μLO (Thérien et al., 2024) — Maximal Update Parametrization for learned optimizers, improving width generalization
- PyLO (Janson et al., 2025) — PyTorch-accessible learned optimizers with CUDA acceleration
- Antoniou et al. (2023) — evaluating whether scaling learned optimizers is worth it

**Graph MetaNetworks (Direct Precursors):**
- Lim et al. (2023) — *Graph Metanetworks for Processing Diverse Neural Architectures* — core GMN framework processing NN weight spaces as graphs
- Kofinas et al. (2024) — *GNN for Learning Equivariant Representations of NNs* — computational graph representation preserving permutation symmetry
- Kalogeropoulos et al. (2024) — *Scale Equivariant Graph MetaNetworks* — extends GMN with scaling symmetries (ScaleGMN)

**Applications:**
- MAML (Finn et al., 2017) — model-agnostic meta-learning
- OPEN (Goldie et al., 2024) — learned optimization for RL addressing plasticity/exploration
- Bihlo (2023) — meta-learned optimization for physics-informed neural networks
- MADA (Ozkara et al., 2024) — meta-adaptive optimizers via hyper-gradient descent

### Current State of the Art

1. **Best practical learned optimizer**: Celo (2025) — achieves strong meta-generalization with 24 GPU hours of meta-training
2. **Largest scale**: VeLO (2022) — 4000 TPU-months but strongest absolute performance
3. **Graph-based NN processing**: Kofinas et al. (2024) and Kalogeropoulos et al. (2024) — process NNs as graphs with equivariance, applied to classification/editing but not yet fully exploited for optimization
4. **Key open problem**: Combining the graph metanetwork paradigm (which respects NN symmetries) with learned optimization (which needs to generalize across architectures)

---

## Research Plan: Graph MetaNetwork Learned Optimizers

### Core Hypothesis

By processing the target neural network's computation graph with a GNN, the learned optimizer can:
1. **Generalize across architectures** — a single optimizer handles MLPs, CNNs, Transformers
2. **Exploit structural information** — layer connectivity and parameter topology inform update rules
3. **Respect symmetries** — permutation equivariance of the GNN preserves weight-space symmetries

### Phase 1: Foundation (Implemented)

- [x] Neural network → computation graph conversion (`nn_to_graph`)
- [x] Edge-centric message passing GNN layers (`gnn_layers.py`)
- [x] Update head architectures (uniform and per-parameter-type)
- [x] GraphMetaOptimizer combining graph processing with update generation
- [x] Per-edge hidden states (GRU-based) for momentum-like behavior
- [x] Meta-training loop with truncated BPTT
- [x] Task library (quadratic, Rosenbrock, MLP regression/classification, CNN)
- [x] YAML-based configuration system
- [x] W&B integration for experiment monitoring

### Phase 2: Architecture Ablations (Planned)

- [ ] Vary GNN depth (1-5 layers) and width (32-256 hidden dim)
- [ ] Compare message-passing rounds (1 vs 2 vs 4)
- [ ] Evaluate per-parameter-type heads vs uniform heads
- [ ] Test with/without hidden states
- [ ] Explore attention-based message passing (Graph Transformer layers)
- [ ] Add scale equivariance following ScaleGMN insights

### Phase 3: Meta-Generalization (Planned)

- [ ] Meta-train on small tasks, evaluate on larger/deeper/wider networks
- [ ] Cross-architecture evaluation (train on MLP, evaluate on CNN)
- [ ] Incorporate μP-style parametrization for width transfer
- [ ] Compare against Adam, SGD, and Celo baselines
- [ ] Horizon generalization: train on short unrolls, evaluate on long training runs

### Phase 4: Advanced Features (Planned)

- [ ] Incorporate gradient history as temporal edge features
- [ ] Explore heterogeneous message passing for different layer types
- [ ] Add loss landscape features (Hessian diagonal approximations)
- [ ] Investigate curriculum meta-training (easy → hard tasks)
- [ ] Scale to real datasets (MNIST, CIFAR-10, small NLP tasks)

---

## Codebase Organization

```
graph_metanet/
├── __init__.py              # Package init
├── train.py                 # Main training script
├── configs/
│   ├── __init__.py
│   ├── config.py            # Dataclass configs + YAML loading
│   ├── base.yaml            # Default configuration
│   ├── experiment_quadratic.yaml
│   └── experiment_multitask.yaml
├── data/
│   ├── __init__.py
│   └── nn_to_graph.py       # NN → computation graph conversion
├── models/
│   ├── __init__.py
│   ├── gnn_layers.py        # Message-passing GNN layers
│   ├── update_heads.py      # Update decoding heads
│   └── learned_optimizer.py # Main GraphMetaOptimizer
├── meta_training/
│   ├── __init__.py
│   └── trainer.py           # Inner/outer loop training
├── tasks/
│   ├── __init__.py
│   └── task_library.py      # Target task definitions
├── tests/
│   ├── __init__.py
│   ├── test_nn_to_graph.py
│   ├── test_models.py
│   ├── test_tasks.py
│   ├── test_config.py
│   └── test_trainer.py
└── utils/
    └── __init__.py
```

### Running Experiments

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest graph_metanet/tests/ -v

# Train with base config
python -m graph_metanet.train --config graph_metanet/configs/base.yaml

# Train with specific experiment
python -m graph_metanet.train --config graph_metanet/configs/experiment_quadratic.yaml

# Override config values
python -m graph_metanet.train --config graph_metanet/configs/base.yaml \
    --overrides '{"seed": 123, "model": {"num_gnn_layers": 5}}'
```
