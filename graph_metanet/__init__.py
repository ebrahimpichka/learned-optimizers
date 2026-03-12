"""Graph MetaNetwork Learned Optimizers.

A research framework for using Graph Neural Networks as learned optimizers,
where the computation graph of the target neural network is processed by a GNN
to produce parameter updates.

Key components:
- nn_to_graph: Convert neural network architectures to computation graphs
- models: GNN-based metanetwork architectures for learned optimization
- meta_training: Inner/outer loop meta-training procedures
- tasks: Target tasks for meta-training and evaluation
"""

__version__ = "0.1.0"
