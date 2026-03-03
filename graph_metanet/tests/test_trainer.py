"""Tests for the meta-training loop."""

import torch
import torch.nn as nn
import pytest

from graph_metanet.models.learned_optimizer import GraphMetaOptimizer
from graph_metanet.tasks.task_library import build_task
from graph_metanet.meta_training.trainer import (
    meta_train_step,
    evaluate_optimizer,
    inner_loop_step,
)
from graph_metanet.data.nn_to_graph import nn_to_graph, build_optimizer_features


class TestInnerLoopStep:
    def test_step_returns_updates(self):
        target = nn.Linear(5, 3)
        opt = GraphMetaOptimizer(
            node_hidden_dim=16, edge_hidden_dim=16, gnn_hidden_dim=32,
            num_gnn_layers=1, num_mp_rounds=1, hidden_state_dim=8,
        )

        # Compute a gradient
        x = torch.randn(2, 5)
        loss = target(x).sum()
        loss.backward()

        graph = nn_to_graph(target)

        new_hidden, updates = inner_loop_step(
            target, opt, graph, loss.item(), step=0, hidden_state=None
        )

        # Should return updates for all named parameters
        for name, _ in target.named_parameters():
            assert name in updates, f"Missing update for {name}"

        # Hidden state should be returned
        assert new_hidden is not None


class TestMetaTrainStep:
    def test_returns_metrics(self):
        opt = GraphMetaOptimizer(
            node_hidden_dim=16, edge_hidden_dim=16, gnn_hidden_dim=32,
            num_gnn_layers=1, num_mp_rounds=1, hidden_state_dim=8,
        )
        meta_opt = torch.optim.Adam(opt.parameters(), lr=1e-3)
        task = build_task("quadratic", dim=5, condition_number=5.0)

        metrics = meta_train_step(
            optimizer=opt,
            meta_optimizer=meta_opt,
            task_spec=task,
            num_inner_steps=5,
            truncation_length=5,
        )

        assert "inner_loss_start" in metrics
        assert "inner_loss_end" in metrics
        assert "meta_loss" in metrics

    def test_mlp_regression_task(self):
        opt = GraphMetaOptimizer(
            node_hidden_dim=16, edge_hidden_dim=16, gnn_hidden_dim=32,
            num_gnn_layers=1, num_mp_rounds=1, hidden_state_dim=8,
        )
        meta_opt = torch.optim.Adam(opt.parameters(), lr=1e-3)
        task = build_task("mlp_regression", input_dim=5, hidden_dim=8,
                          output_dim=1, num_layers=2)

        metrics = meta_train_step(
            optimizer=opt,
            meta_optimizer=meta_opt,
            task_spec=task,
            num_inner_steps=3,
            truncation_length=3,
        )
        assert isinstance(metrics["meta_loss"], float)


class TestEvaluateOptimizer:
    def test_returns_loss_curve(self):
        opt = GraphMetaOptimizer(
            node_hidden_dim=16, edge_hidden_dim=16, gnn_hidden_dim=32,
            num_gnn_layers=1, num_mp_rounds=1, hidden_state_dim=8,
        )
        task = build_task("quadratic", dim=5, condition_number=5.0)

        losses = evaluate_optimizer(opt, task, num_steps=5)
        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)
