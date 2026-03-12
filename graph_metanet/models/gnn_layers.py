"""GNN message-passing layers for processing neural network graphs.

Implements edge-centric message passing suitable for the graph metanetwork
paradigm, where edge features encode parameter information and node features
encode layer/operation information.
"""

import torch
import torch.nn as nn
from typing import Optional


class EdgeMessagePassing(nn.Module):
    """Single round of edge-centric message passing.

    Computes messages from edge features and incident node features,
    then aggregates to update both node and edge representations.

    Args:
        node_dim: Dimensionality of node features.
        edge_dim: Dimensionality of edge features.
        hidden_dim: Hidden layer size in message MLPs.
        use_layer_norm: Whether to apply LayerNorm after updates.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        # Edge update MLP: combines src node, dst node, and current edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        # Node update MLP: combines current node features with aggregated messages
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.edge_norm = nn.LayerNorm(edge_dim)
            self.node_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple:
        """Forward pass.

        Args:
            node_features: (num_nodes, node_dim)
            edge_index: (2, num_edges) - [src, dst]
            edge_features: (num_edges, edge_dim)

        Returns:
            Tuple of updated (node_features, edge_features).
        """
        src, dst = edge_index[0], edge_index[1]

        # Edge update
        src_feats = node_features[src]  # (num_edges, node_dim)
        dst_feats = node_features[dst]  # (num_edges, node_dim)
        edge_input = torch.cat([src_feats, dst_feats, edge_features], dim=-1)
        edge_update = self.edge_mlp(edge_input)
        edge_features = edge_features + edge_update  # residual
        if self.use_layer_norm:
            edge_features = self.edge_norm(edge_features)

        # Node update: aggregate incoming edge messages (mean)
        num_nodes = node_features.shape[0]
        agg = torch.zeros(num_nodes, edge_features.shape[-1],
                          device=node_features.device)
        count = torch.zeros(num_nodes, 1, device=node_features.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(edge_features), edge_features)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones_like(dst.unsqueeze(-1).float()))
        count = count.clamp(min=1)
        agg = agg / count

        node_input = torch.cat([node_features, agg], dim=-1)
        node_update = self.node_mlp(node_input)
        node_features = node_features + node_update  # residual
        if self.use_layer_norm:
            node_features = self.node_norm(node_features)

        return node_features, edge_features


class MetaGNNLayer(nn.Module):
    """A full GNN block for the graph metanetwork.

    Stacks multiple rounds of edge message passing with optional
    skip connections.

    Args:
        node_dim: Node feature dimensionality.
        edge_dim: Edge feature dimensionality.
        hidden_dim: Hidden size for message MLPs.
        num_rounds: Number of message-passing rounds per layer.
        use_layer_norm: Whether to use LayerNorm.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_rounds: int = 2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.rounds = nn.ModuleList([
            EdgeMessagePassing(node_dim, edge_dim, hidden_dim, use_layer_norm)
            for _ in range(num_rounds)
        ])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple:
        for mp_round in self.rounds:
            node_features, edge_features = mp_round(
                node_features, edge_index, edge_features
            )
        return node_features, edge_features
