"""Tests for neural network to graph conversion."""

import torch
import torch.nn as nn
import pytest

from graph_metanet.data.nn_to_graph import (
    nn_to_graph,
    build_optimizer_features,
    NNGraph,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
)


class SimpleLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.fc1 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = x.mean(dim=[2, 3])
        return self.fc1(x)


class TestNNToGraph:
    def test_basic_linear_net(self):
        model = SimpleLinearNet()
        graph = nn_to_graph(model)

        assert isinstance(graph, NNGraph)
        # 2 linear layers + input + output = 4 nodes
        assert graph.node_features.shape[0] == 4
        assert graph.node_features.shape[1] == NODE_FEAT_DIM

    def test_edge_count(self):
        model = SimpleLinearNet()
        graph = nn_to_graph(model)

        # fc1.weight, fc1.bias, fc2.weight, fc2.bias + output edge = 5
        assert graph.edge_index.shape[1] == 5
        assert graph.edge_features.shape[0] == 5

    def test_param_refs(self):
        model = SimpleLinearNet()
        graph = nn_to_graph(model)

        # Check that parameter references are correctly populated
        refs = graph.param_refs
        param_names = [(r[0], r[1]) for r in refs if r[0] != "__output__"]
        assert ("fc1", "weight") in param_names
        assert ("fc1", "bias") in param_names
        assert ("fc2", "weight") in param_names
        assert ("fc2", "bias") in param_names

    def test_param_shapes(self):
        model = SimpleLinearNet()
        graph = nn_to_graph(model)

        # fc1.weight shape should be (20, 10)
        weight_idx = None
        for i, ref in enumerate(graph.param_refs):
            if ref == ("fc1", "weight"):
                weight_idx = i
                break
        assert weight_idx is not None
        assert graph.param_shapes[weight_idx] == torch.Size([20, 10])

    def test_cnn_conversion(self):
        model = SimpleCNN()
        graph = nn_to_graph(model)

        assert graph.node_features.shape[0] == 4  # input + conv1 + fc1 + output
        assert graph.edge_index.shape[0] == 2

    def test_edge_features_shape(self):
        model = SimpleLinearNet()
        graph = nn_to_graph(model)

        assert graph.edge_features.shape[1] == EDGE_FEAT_DIM

    def test_empty_model(self):
        model = nn.Sequential()  # No parameters
        graph = nn_to_graph(model)

        # input + output nodes, no edges
        assert graph.node_features.shape[0] == 2
        assert graph.edge_features.shape[0] == 0

    def test_build_optimizer_features(self):
        model = SimpleLinearNet()
        graph = nn_to_graph(model)

        # Without gradients
        feats = build_optimizer_features(model, graph, step=5, loss_val=1.0)
        assert feats.shape[0] == graph.edge_features.shape[0]

    def test_build_optimizer_features_with_grads(self):
        model = SimpleLinearNet()
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        graph = nn_to_graph(model)
        grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        feats = build_optimizer_features(model, graph, grads=grads, step=1, loss_val=loss.item())
        assert feats.shape[0] == graph.edge_features.shape[0]
