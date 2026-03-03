"""Simple target tasks for meta-training and evaluating learned optimizers.

Each task defines:
- A neural network architecture (the target model)
- A data generator or dataset loader
- A loss function

Tasks are designed to be lightweight for meta-training while covering
diverse optimization landscapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TaskSpec:
    """Specification of a meta-training task.

    Attributes:
        name: Human-readable task name.
        model_fn: Callable returning a fresh target model instance.
        data_fn: Callable returning (input, target) batch tensors.
        loss_fn: Loss function taking (model_output, target) -> scalar.
        num_steps: Default number of inner-loop optimization steps.
    """
    name: str
    model_fn: Callable[[], nn.Module]
    data_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    num_steps: int = 100


def _make_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2
) -> nn.Module:
    layers = []
    prev_dim = input_dim
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _make_small_cnn(num_classes: int = 10) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, num_classes),
    )


# --- Quadratic task ---

class QuadraticTask:
    """Minimization of a random quadratic: f(x) = 0.5 * x^T A x + b^T x."""

    def __init__(self, dim: int = 10, condition_number: float = 10.0):
        self.dim = dim
        self.condition_number = condition_number

    def build(self) -> TaskSpec:
        dim = self.dim
        kappa = self.condition_number

        def model_fn():
            model = nn.Linear(dim, 1, bias=False)
            nn.init.normal_(model.weight, std=0.5)
            return model

        def data_fn():
            # Random PSD matrix with controlled condition number
            eigenvalues = torch.linspace(1.0, kappa, dim)
            Q = torch.randn(dim, dim)
            Q, _ = torch.linalg.qr(Q)
            A = Q @ torch.diag(eigenvalues) @ Q.T
            b = torch.randn(dim)
            return A, b

        def loss_fn(output, target):
            # target = (A, b); output = model.weight (1, dim)
            A, b = target
            x = output.squeeze()
            return 0.5 * x @ A @ x + b @ x

        return TaskSpec(
            name=f"quadratic_d{dim}_k{kappa}",
            model_fn=model_fn,
            data_fn=data_fn,
            loss_fn=loss_fn,
            num_steps=50,
        )


# --- Rosenbrock task ---

class RosenbrockTask:
    """Minimize the Rosenbrock function using a parametric model."""

    def __init__(self, dim: int = 2):
        self.dim = dim

    def build(self) -> TaskSpec:
        dim = self.dim

        def model_fn():
            model = nn.Linear(dim, 1, bias=False)
            nn.init.uniform_(model.weight, -2.0, 2.0)
            return model

        def data_fn():
            return torch.empty(0), torch.empty(0)

        def loss_fn(output, _target):
            x = output.squeeze()
            loss = torch.tensor(0.0)
            for i in range(dim - 1):
                loss = loss + 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
            return loss

        return TaskSpec(
            name=f"rosenbrock_d{dim}",
            model_fn=model_fn,
            data_fn=data_fn,
            loss_fn=loss_fn,
            num_steps=200,
        )


# --- MLP regression task ---

class MLPRegressionTask:
    """Train a small MLP on random regression data."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 32,
        output_dim: int = 1,
        num_layers: int = 2,
        num_samples: int = 64,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_samples = num_samples

    def build(self) -> TaskSpec:
        cfg = self

        def model_fn():
            return _make_mlp(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.num_layers)

        def data_fn():
            x = torch.randn(cfg.num_samples, cfg.input_dim)
            # Random linear target with noise
            w = torch.randn(cfg.input_dim, cfg.output_dim)
            y = x @ w + 0.1 * torch.randn(cfg.num_samples, cfg.output_dim)
            return x, y

        return TaskSpec(
            name=f"mlp_reg_{cfg.input_dim}_{cfg.hidden_dim}x{cfg.num_layers}",
            model_fn=model_fn,
            data_fn=data_fn,
            loss_fn=lambda pred, target: F.mse_loss(pred, target),
            num_steps=100,
        )


# --- MLP classification task ---

class MLPClassificationTask:
    """Train a small MLP on random classification data."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 32,
        num_classes: int = 3,
        num_layers: int = 2,
        num_samples: int = 128,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_samples = num_samples

    def build(self) -> TaskSpec:
        cfg = self

        def model_fn():
            return _make_mlp(cfg.input_dim, cfg.hidden_dim, cfg.num_classes, cfg.num_layers)

        def data_fn():
            x = torch.randn(cfg.num_samples, cfg.input_dim)
            y = torch.randint(0, cfg.num_classes, (cfg.num_samples,))
            return x, y

        return TaskSpec(
            name=f"mlp_cls_{cfg.input_dim}_{cfg.hidden_dim}x{cfg.num_layers}_{cfg.num_classes}c",
            model_fn=model_fn,
            data_fn=data_fn,
            loss_fn=lambda pred, target: F.cross_entropy(pred, target),
            num_steps=100,
        )


# --- Small CNN task ---

class SmallCNNTask:
    """Train a small CNN on random image-like classification data."""

    def __init__(
        self,
        num_classes: int = 10,
        num_samples: int = 64,
        image_size: int = 16,
    ):
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image_size = image_size

    def build(self) -> TaskSpec:
        cfg = self

        def model_fn():
            return _make_small_cnn(cfg.num_classes)

        def data_fn():
            x = torch.randn(cfg.num_samples, 1, cfg.image_size, cfg.image_size)
            y = torch.randint(0, cfg.num_classes, (cfg.num_samples,))
            return x, y

        return TaskSpec(
            name=f"small_cnn_{cfg.num_classes}c",
            model_fn=model_fn,
            data_fn=data_fn,
            loss_fn=lambda pred, target: F.cross_entropy(pred, target),
            num_steps=100,
        )


# Registry of available task builders
TASK_REGISTRY: Dict[str, Callable[..., TaskSpec]] = {
    "quadratic": lambda **kw: QuadraticTask(**kw).build(),
    "rosenbrock": lambda **kw: RosenbrockTask(**kw).build(),
    "mlp_regression": lambda **kw: MLPRegressionTask(**kw).build(),
    "mlp_classification": lambda **kw: MLPClassificationTask(**kw).build(),
    "small_cnn": lambda **kw: SmallCNNTask(**kw).build(),
}


def build_task(name: str, **kwargs) -> TaskSpec:
    """Build a task by name from the registry.

    Args:
        name: Task name (e.g., "quadratic", "mlp_regression").
        **kwargs: Task-specific keyword arguments.

    Returns:
        A TaskSpec instance.
    """
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name](**kwargs)
