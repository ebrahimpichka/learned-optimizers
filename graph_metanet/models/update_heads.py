"""Update head modules that decode GNN edge representations into parameter updates.

Supports multiple head architectures for flexibility in research:
- UpdateHead: simple MLP per-edge
- PerParamUpdateHead: separate heads per parameter type
"""

import torch
import torch.nn as nn
from typing import List, Optional


class UpdateHead(nn.Module):
    """Decodes edge representations into scalar or vector parameter updates.

    Args:
        edge_dim: Input edge feature dimensionality.
        hidden_dim: Hidden layer size.
        output_scale: Initial scale for the output (controls step size magnitude).
    """

    def __init__(self, edge_dim: int, hidden_dim: int, output_scale: float = 0.001):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.output_scale = output_scale
        # Initialize last layer to near-zero for stable start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """Produce per-edge scalar updates.

        Args:
            edge_features: (num_edges, edge_dim)

        Returns:
            Tensor of shape (num_edges, 1) with update magnitudes.
        """
        return self.mlp(edge_features) * self.output_scale


class PerParamUpdateHead(nn.Module):
    """Separate update heads for different parameter categories.

    Uses distinct MLPs for weights vs biases, allowing the optimizer
    to learn different update strategies for each.

    Args:
        edge_dim: Input edge feature dimensionality.
        hidden_dim: Hidden layer size.
        output_scale: Initial output scale.
    """

    def __init__(self, edge_dim: int, hidden_dim: int, output_scale: float = 0.001):
        super().__init__()
        self.weight_head = UpdateHead(edge_dim, hidden_dim, output_scale)
        self.bias_head = UpdateHead(edge_dim, hidden_dim, output_scale)

    def forward(
        self,
        edge_features: torch.Tensor,
        is_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce per-edge updates, routing through weight or bias heads.

        Args:
            edge_features: (num_edges, edge_dim)
            is_bias: Boolean tensor of shape (num_edges,) indicating bias params.
                If None, all edges use the weight head.

        Returns:
            Tensor of shape (num_edges, 1).
        """
        if is_bias is None:
            return self.weight_head(edge_features)

        updates = torch.zeros(edge_features.shape[0], 1,
                              device=edge_features.device)
        bias_mask = is_bias.bool()
        weight_mask = ~bias_mask

        if weight_mask.any():
            updates[weight_mask] = self.weight_head(edge_features[weight_mask])
        if bias_mask.any():
            updates[bias_mask] = self.bias_head(edge_features[bias_mask])

        return updates
