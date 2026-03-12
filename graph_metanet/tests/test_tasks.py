"""Tests for the task library."""

import torch
import torch.nn as nn
import pytest

from graph_metanet.tasks.task_library import (
    build_task,
    QuadraticTask,
    RosenbrockTask,
    MLPRegressionTask,
    MLPClassificationTask,
    SmallCNNTask,
    TASK_REGISTRY,
)


class TestTaskRegistry:
    def test_all_tasks_registered(self):
        expected = ["quadratic", "rosenbrock", "mlp_regression",
                    "mlp_classification", "small_cnn"]
        for name in expected:
            assert name in TASK_REGISTRY

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            build_task("nonexistent_task")


class TestQuadraticTask:
    def test_build(self):
        spec = QuadraticTask(dim=5, condition_number=10.0).build()
        assert spec.name == "quadratic_d5_k10.0"
        model = spec.model_fn()
        assert isinstance(model, nn.Module)

    def test_loss_computes(self):
        spec = QuadraticTask(dim=5).build()
        model = spec.model_fn()
        data_x, data_y = spec.data_fn()
        output = list(model.parameters())[0]
        loss = spec.loss_fn(output, (data_x, data_y))
        assert loss.ndim == 0  # scalar


class TestRosenbrockTask:
    def test_build(self):
        spec = RosenbrockTask(dim=3).build()
        model = spec.model_fn()
        assert isinstance(model, nn.Module)

    def test_loss_at_optimum(self):
        spec = RosenbrockTask(dim=2).build()
        model = spec.model_fn()
        # Set parameters to the optimum (1, 1)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 1.0]]))
        output = model.weight
        loss = spec.loss_fn(output, torch.empty(0))
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestMLPRegressionTask:
    def test_build(self):
        spec = MLPRegressionTask(input_dim=5, hidden_dim=16, num_layers=2).build()
        model = spec.model_fn()
        data_x, data_y = spec.data_fn()
        output = model(data_x)
        loss = spec.loss_fn(output, data_y)
        assert loss.ndim == 0

    def test_data_shapes(self):
        spec = MLPRegressionTask(input_dim=5, num_samples=32).build()
        data_x, data_y = spec.data_fn()
        assert data_x.shape == (32, 5)


class TestMLPClassificationTask:
    def test_build(self):
        spec = MLPClassificationTask(input_dim=8, num_classes=4).build()
        model = spec.model_fn()
        data_x, data_y = spec.data_fn()
        output = model(data_x)
        loss = spec.loss_fn(output, data_y)
        assert loss.ndim == 0


class TestSmallCNNTask:
    def test_build(self):
        spec = SmallCNNTask(num_classes=5, image_size=8).build()
        model = spec.model_fn()
        data_x, data_y = spec.data_fn()
        output = model(data_x)
        loss = spec.loss_fn(output, data_y)
        assert loss.ndim == 0
