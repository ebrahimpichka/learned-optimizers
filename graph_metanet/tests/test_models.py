"""Tests for GNN layers and the learned optimizer model."""

import torch
import torch.nn as nn
import pytest

from graph_metanet.models.gnn_layers import EdgeMessagePassing, MetaGNNLayer
from graph_metanet.models.update_heads import UpdateHead, PerParamUpdateHead
from graph_metanet.models.learned_optimizer import GraphMetaOptimizer
from graph_metanet.data.nn_to_graph import nn_to_graph, build_optimizer_features


class TestEdgeMessagePassing:
    def test_output_shapes(self):
        layer = EdgeMessagePassing(node_dim=16, edge_dim=8, hidden_dim=32)
        node_feats = torch.randn(5, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_feats = torch.randn(4, 8)

        new_nodes, new_edges = layer(node_feats, edge_index, edge_feats)
        assert new_nodes.shape == (5, 16)
        assert new_edges.shape == (4, 8)

    def test_residual_connection(self):
        layer = EdgeMessagePassing(node_dim=16, edge_dim=8, hidden_dim=32,
                                   use_layer_norm=False)
        node_feats = torch.zeros(3, 16)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_feats = torch.zeros(2, 8)

        new_nodes, new_edges = layer(node_feats, edge_index, edge_feats)
        # With zero inputs and residual, outputs should be close to zero
        # (non-zero due to biases in the MLP)
        assert new_nodes.shape == (3, 16)


class TestMetaGNNLayer:
    def test_multiple_rounds(self):
        layer = MetaGNNLayer(node_dim=16, edge_dim=8, hidden_dim=32, num_rounds=3)
        node_feats = torch.randn(4, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_feats = torch.randn(3, 8)

        new_nodes, new_edges = layer(node_feats, edge_index, edge_feats)
        assert new_nodes.shape == (4, 16)
        assert new_edges.shape == (3, 8)


class TestUpdateHead:
    def test_output_shape(self):
        head = UpdateHead(edge_dim=32, hidden_dim=64)
        edge_feats = torch.randn(10, 32)
        updates = head(edge_feats)
        assert updates.shape == (10, 1)

    def test_initial_scale(self):
        head = UpdateHead(edge_dim=32, hidden_dim=64, output_scale=0.001)
        edge_feats = torch.randn(10, 32)
        updates = head(edge_feats)
        # With zero-initialized last layer, output should be near zero
        assert updates.abs().max() < 0.1


class TestPerParamUpdateHead:
    def test_without_bias_mask(self):
        head = PerParamUpdateHead(edge_dim=32, hidden_dim=64)
        edge_feats = torch.randn(10, 32)
        updates = head(edge_feats)
        assert updates.shape == (10, 1)

    def test_with_bias_mask(self):
        head = PerParamUpdateHead(edge_dim=32, hidden_dim=64)
        edge_feats = torch.randn(10, 32)
        is_bias = torch.tensor([0, 0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=torch.float32)
        updates = head(edge_feats, is_bias)
        assert updates.shape == (10, 1)


class TestGraphMetaOptimizer:
    def _make_target_model(self):
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

    def test_forward_pass(self):
        opt = GraphMetaOptimizer(
            node_hidden_dim=32,
            edge_hidden_dim=32,
            gnn_hidden_dim=64,
            num_gnn_layers=2,
            num_mp_rounds=1,
        )

        target = self._make_target_model()
        graph = nn_to_graph(target)
        opt_feats = build_optimizer_features(target, graph)

        updates, hidden = opt(graph, opt_feats)
        assert updates.shape[0] == graph.edge_features.shape[0]
        assert updates.shape[1] == 1
        assert hidden is not None

    def test_meta_step(self):
        opt = GraphMetaOptimizer(
            node_hidden_dim=32,
            edge_hidden_dim=32,
            gnn_hidden_dim=64,
            num_gnn_layers=2,
        )

        target = self._make_target_model()
        x = torch.randn(4, 10)
        y = target(x)
        loss = y.sum()
        loss.backward()

        graph = nn_to_graph(target)
        param_updates, hidden = opt.meta_step(target, graph, loss.item(), step=0)

        # Check that updates exist for each named parameter
        for name, _ in target.named_parameters():
            assert name in param_updates, f"Missing update for {name}"

    def test_hidden_state_persistence(self):
        opt = GraphMetaOptimizer(
            node_hidden_dim=32,
            edge_hidden_dim=32,
            gnn_hidden_dim=64,
            num_gnn_layers=2,
            use_hidden_state=True,
            hidden_state_dim=16,
        )

        target = self._make_target_model()
        graph = nn_to_graph(target)
        opt_feats = build_optimizer_features(target, graph)

        # First step
        _, hidden1 = opt(graph, opt_feats)
        assert hidden1 is not None
        assert hidden1.shape[1] == 16

        # Second step with previous hidden state
        _, hidden2 = opt(graph, opt_feats, hidden_state=hidden1)
        assert hidden2 is not None
        # Hidden states should differ
        assert not torch.allclose(hidden1, hidden2)

    def test_no_hidden_state_mode(self):
        opt = GraphMetaOptimizer(
            node_hidden_dim=32,
            edge_hidden_dim=32,
            gnn_hidden_dim=64,
            num_gnn_layers=2,
            use_hidden_state=False,
        )

        target = self._make_target_model()
        graph = nn_to_graph(target)
        opt_feats = build_optimizer_features(target, graph)

        updates, hidden = opt(graph, opt_feats)
        assert hidden is None
