"""Meta-training loop for learned optimizers.

Implements the inner/outer loop training procedure where:
- Inner loop: The learned optimizer updates the target model parameters
- Outer loop: The learned optimizer itself is updated based on the
  target model's performance

Uses functional parameter tracking and truncated backpropagation through
time (TBPTT) to enable differentiable inner-loop updates while managing
memory during long unrolls.
"""

import torch
import torch.nn as nn
from torch.func import functional_call
from typing import Dict, List, Optional, Tuple
import logging

from graph_metanet.data.nn_to_graph import nn_to_graph, build_optimizer_features, NNGraph
from graph_metanet.models.learned_optimizer import GraphMetaOptimizer
from graph_metanet.tasks.task_library import TaskSpec

logger = logging.getLogger(__name__)


def _is_function_opt_task(name: str) -> bool:
    return name.startswith("quadratic") or name.startswith("rosenbrock")


def _compute_loss(task_spec, target_model, params, data_x, data_y):
    """Compute loss using functional_call for non-function-opt tasks."""
    if _is_function_opt_task(task_spec.name):
        output = list(params.values())[0]
        return task_spec.loss_fn(output, (data_x, data_y))
    else:
        output = functional_call(target_model, params, (data_x,))
        return task_spec.loss_fn(output, data_y)


def inner_loop_step(
    target_model: nn.Module,
    optimizer: GraphMetaOptimizer,
    graph: NNGraph,
    loss_val: float,
    step: int,
    hidden_state: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Execute one inner-loop optimization step (non-differentiable, for evaluation).

    Args:
        target_model: Target model being optimized.
        optimizer: The learned optimizer (GraphMetaOptimizer).
        graph: Computation graph of the target model.
        loss_val: Current loss value.
        step: Inner-loop step index.
        hidden_state: Per-edge hidden state from previous step.

    Returns:
        Tuple of (new_hidden_state, param_updates_dict).
    """
    param_updates, new_hidden = optimizer.meta_step(
        target_model, graph, loss_val, step, hidden_state
    )

    # Apply updates to target model parameters
    with torch.no_grad():
        for name, param in target_model.named_parameters():
            if name in param_updates:
                param.data.add_(param_updates[name])

    return new_hidden, param_updates


def meta_train_step(
    optimizer: GraphMetaOptimizer,
    meta_optimizer: torch.optim.Optimizer,
    task_spec: TaskSpec,
    num_inner_steps: int,
    truncation_length: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Execute one meta-training step (outer loop iteration).

    Uses functional parameter tracking so that inner-loop updates are
    differentiable with respect to the learned optimizer's parameters.
    Truncated BPTT segments the unroll to manage memory.

    Args:
        optimizer: The learned optimizer to meta-train.
        meta_optimizer: Optimizer for the learned optimizer's parameters.
        task_spec: The task specification.
        num_inner_steps: Number of inner-loop steps.
        truncation_length: Length of truncation segments for BPTT.
        device: Compute device.

    Returns:
        Dict with metrics: inner_loss_start, inner_loss_end, meta_loss.
    """
    optimizer.train()

    # Fresh target model
    target_model = task_spec.model_fn().to(device)
    data_x, data_y = task_spec.data_fn()
    data_x, data_y = data_x.to(device), data_y.to(device)

    # Build computation graph (structure doesn't change across steps)
    graph = nn_to_graph(target_model)

    # Initialize params as detached leaf tensors
    params = {
        name: p.detach().clone().requires_grad_(True)
        for name, p in target_model.named_parameters()
    }

    hidden_state = None
    segment_meta_loss = torch.tensor(0.0, device=device)
    inner_losses = []

    for step in range(num_inner_steps):
        # Compute loss with current params
        loss = _compute_loss(task_spec, target_model, params, data_x, data_y)
        inner_losses.append(loss.item())

        # Compute target gradients (retain graph so meta_loss backward works)
        param_list = list(params.values())
        grads = torch.autograd.grad(
            loss, param_list, create_graph=False,
            retain_graph=True, allow_unused=True,
        )
        grad_dict = {}
        for (name, _), g in zip(params.items(), grads):
            if g is not None:
                grad_dict[name] = g.detach()

        # Build optimizer features
        opt_feats = build_optimizer_features(
            target_model, graph, grad_dict, step, loss.item()
        ).to(device)

        # Determine bias mask
        is_bias = torch.tensor(
            [ref[1] == "bias" for ref in graph.param_refs],
            device=device,
        ).float()

        # Run learned optimizer (differentiable w.r.t. optimizer params)
        if hidden_state is not None:
            hidden_state = hidden_state.detach()
        updates_raw, hidden_state = optimizer(
            graph, opt_feats, hidden_state, is_bias
        )

        # Apply updates differentiably to create new params
        new_params = {}
        for name in params:
            matched = False
            for i, (mod_name, p_name) in enumerate(graph.param_refs):
                full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                if full_name == name:
                    grad = grad_dict.get(name)
                    if grad is not None:
                        direction = -grad
                    else:
                        direction = torch.zeros_like(params[name])
                    # updates_raw[i, 0] is differentiable w.r.t. optimizer
                    new_params[name] = params[name].detach() + updates_raw[i, 0] * direction
                    matched = True
                    break
            if not matched:
                new_params[name] = params[name].detach()

        params = new_params

        # Accumulate post-update loss into segment meta-loss
        post_loss = _compute_loss(task_spec, target_model, params, data_x, data_y)
        segment_meta_loss = segment_meta_loss + post_loss

        # Truncated BPTT: backprop at segment boundaries
        if (step + 1) % truncation_length == 0 or step == num_inner_steps - 1:
            meta_optimizer.zero_grad()
            segment_meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer.parameters(), max_norm=1.0)
            meta_optimizer.step()

            # Detach everything for next segment
            params = {
                n: p.detach().clone().requires_grad_(True)
                for n, p in params.items()
            }
            if hidden_state is not None:
                hidden_state = hidden_state.detach()
            segment_meta_loss = torch.tensor(0.0, device=device)

    return {
        "inner_loss_start": inner_losses[0] if inner_losses else 0.0,
        "inner_loss_end": inner_losses[-1] if inner_losses else 0.0,
        "meta_loss": sum(inner_losses) / len(inner_losses) if inner_losses else 0.0,
    }


def evaluate_optimizer(
    optimizer: GraphMetaOptimizer,
    task_spec: TaskSpec,
    num_steps: int,
    device: torch.device = torch.device("cpu"),
) -> List[float]:
    """Evaluate a learned optimizer on a task.

    Args:
        optimizer: The learned optimizer (in eval mode).
        task_spec: Task to evaluate on.
        num_steps: Number of optimization steps.
        device: Compute device.

    Returns:
        List of loss values at each step.
    """
    optimizer.eval()

    target_model = task_spec.model_fn().to(device)
    data_x, data_y = task_spec.data_fn()
    data_x, data_y = data_x.to(device), data_y.to(device)

    graph = nn_to_graph(target_model)
    hidden_state = None
    losses = []

    with torch.no_grad():
        for step in range(num_steps):
            # Enable grads temporarily for target model
            with torch.enable_grad():
                for p in target_model.parameters():
                    p.requires_grad_(True)
                target_model.zero_grad()

                if _is_function_opt_task(task_spec.name):
                    output = list(target_model.parameters())[0]
                    loss = task_spec.loss_fn(output, (data_x, data_y))
                else:
                    output = target_model(data_x)
                    loss = task_spec.loss_fn(output, data_y)

                losses.append(loss.item())
                loss.backward()

            hidden_state, _ = inner_loop_step(
                target_model, optimizer, graph, loss.item(), step, hidden_state
            )

    return losses
