"""Graph MetaNetwork Learned Optimizer.

The core optimizer module that:
1. Converts the target network to a computation graph
2. Processes it with a GNN to produce graph-aware representations
3. Decodes per-parameter update directions and magnitudes

Designed following insights from:
- Graph Metanetworks (Lim et al., 2023)
- GNN for Equivariant Representations of NNs (Kofinas et al., 2024)
- Scale Equivariant Graph Metanetworks (Kalogeropoulos et al., 2024)
- VeLO / Celo learned optimizer architectures (Metz et al., 2022; Moudgil et al., 2025)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from graph_metanet.data.nn_to_graph import (
    NNGraph,
    nn_to_graph,
    build_optimizer_features,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
)
from graph_metanet.models.gnn_layers import MetaGNNLayer
from graph_metanet.models.update_heads import UpdateHead, PerParamUpdateHead


class GraphMetaOptimizer(nn.Module):
    """A learned optimizer that uses a GNN on the target network's computation graph.

    Architecture overview:
        1. Input projection: project raw node/edge features into hidden space
        2. GNN backbone: stack of MetaGNNLayer blocks for message passing
        3. Update head: decode edge representations into per-parameter updates

    The optimizer maintains per-parameter hidden states (edge-level) across
    optimization steps, enabling it to learn momentum-like behaviors.

    Args:
        node_input_dim: Raw node feature dimension (default: NODE_FEAT_DIM).
        edge_input_dim: Raw edge feature dimension including optimizer features.
        node_hidden_dim: Hidden dimension for node representations.
        edge_hidden_dim: Hidden dimension for edge representations.
        gnn_hidden_dim: Hidden dimension inside GNN message-passing MLPs.
        num_gnn_layers: Number of stacked MetaGNNLayer blocks.
        num_mp_rounds: Message-passing rounds per GNN layer.
        use_layer_norm: Whether to use LayerNorm in GNN layers.
        use_per_param_heads: Whether to use separate heads for weights/biases.
        output_scale: Initial output magnitude scale.
        use_hidden_state: Whether to maintain per-edge hidden states across steps.
        hidden_state_dim: Dimension of the per-edge hidden state.
    """

    def __init__(
        self,
        node_input_dim: int = NODE_FEAT_DIM,
        edge_input_dim: int = EDGE_FEAT_DIM + 5,  # base + grad stats + step + loss
        node_hidden_dim: int = 64,
        edge_hidden_dim: int = 64,
        gnn_hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_mp_rounds: int = 2,
        use_layer_norm: bool = True,
        use_per_param_heads: bool = False,
        output_scale: float = 0.001,
        use_hidden_state: bool = True,
        hidden_state_dim: int = 32,
    ):
        super().__init__()

        self.use_hidden_state = use_hidden_state
        self.hidden_state_dim = hidden_state_dim

        # Input projections
        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim, node_hidden_dim),
            nn.GELU(),
        )

        actual_edge_input = edge_input_dim
        if use_hidden_state:
            actual_edge_input += hidden_state_dim

        self.edge_proj = nn.Sequential(
            nn.Linear(actual_edge_input, edge_hidden_dim),
            nn.GELU(),
        )

        # GNN backbone
        self.gnn_layers = nn.ModuleList([
            MetaGNNLayer(
                node_dim=node_hidden_dim,
                edge_dim=edge_hidden_dim,
                hidden_dim=gnn_hidden_dim,
                num_rounds=num_mp_rounds,
                use_layer_norm=use_layer_norm,
            )
            for _ in range(num_gnn_layers)
        ])

        # Update head
        if use_per_param_heads:
            self.update_head = PerParamUpdateHead(
                edge_hidden_dim, gnn_hidden_dim, output_scale
            )
        else:
            self.update_head = UpdateHead(
                edge_hidden_dim, gnn_hidden_dim, output_scale
            )
        self.use_per_param_heads = use_per_param_heads

        # Hidden state update
        if use_hidden_state:
            self.hidden_update = nn.GRUCell(edge_hidden_dim, hidden_state_dim)

    def forward(
        self,
        graph: NNGraph,
        optimizer_features: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        is_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute parameter updates from the graph representation.

        Args:
            graph: The NNGraph of the target network.
            optimizer_features: Per-edge optimizer features (num_edges, feat_dim).
            hidden_state: Optional per-edge hidden state from previous step.
            is_bias: Optional boolean mask indicating bias parameters.

        Returns:
            Tuple of (updates, new_hidden_state) where updates has shape
            (num_edges, 1) and hidden_state has shape (num_edges, hidden_state_dim).
        """
        device = optimizer_features.device

        node_feats = graph.node_features.to(device)
        edge_index = graph.edge_index.to(device)

        # Combine optimizer features with hidden state
        if self.use_hidden_state:
            if hidden_state is None:
                hidden_state = torch.zeros(
                    optimizer_features.shape[0], self.hidden_state_dim,
                    device=device,
                )
            edge_input = torch.cat([optimizer_features, hidden_state], dim=-1)
        else:
            edge_input = optimizer_features

        # Project to hidden space
        node_h = self.node_proj(node_feats)
        edge_h = self.edge_proj(edge_input)

        # GNN message passing
        for gnn_layer in self.gnn_layers:
            node_h, edge_h = gnn_layer(node_h, edge_index, edge_h)

        # Decode updates
        if self.use_per_param_heads and is_bias is not None:
            updates = self.update_head(edge_h, is_bias)
        else:
            updates = self.update_head(edge_h)

        # Update hidden state
        new_hidden = None
        if self.use_hidden_state:
            new_hidden = self.hidden_update(edge_h, hidden_state)

        return updates, new_hidden

    def meta_step(
        self,
        target_model: nn.Module,
        graph: NNGraph,
        loss_val: float,
        step: int,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Perform one optimization step on the target model.

        Computes gradients of the target model, builds optimizer features,
        runs the GNN forward pass, and returns the parameter updates.

        Args:
            target_model: The network being optimized.
            graph: Its computation graph representation.
            loss_val: Current loss value (scalar).
            step: Current training step number.
            hidden_state: Per-edge hidden state from previous step.

        Returns:
            Tuple of (param_updates_dict, new_hidden_state) where
            param_updates_dict maps "module.param" names to update tensors.
        """
        # Collect gradients
        grads = {}
        for name, param in target_model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad

        # Build features
        opt_feats = build_optimizer_features(
            target_model, graph, grads, step, loss_val
        )
        device = next(self.parameters()).device
        opt_feats = opt_feats.to(device)

        # Determine bias mask
        is_bias = torch.tensor([
            ref[1] == "bias" for ref in graph.param_refs
        ], device=device).float()

        # Forward pass
        updates, new_hidden = self.forward(
            graph, opt_feats, hidden_state, is_bias
        )

        # Map updates back to parameters
        param_updates = {}
        for i, (mod_name, p_name) in enumerate(graph.param_refs):
            if mod_name == "__output__":
                continue
            full_name = f"{mod_name}.{p_name}" if mod_name else p_name
            shape = graph.param_shapes[i]
            # Broadcast scalar update to full parameter shape
            param_updates[full_name] = updates[i, 0].expand(shape)

        return param_updates, new_hidden
