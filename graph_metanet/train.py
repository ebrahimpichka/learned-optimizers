"""Main training script for Graph MetaNetwork Learned Optimizer.

Usage:
    python -m graph_metanet.train --config graph_metanet/configs/base.yaml
    python -m graph_metanet.train --config graph_metanet/configs/experiment_quadratic.yaml

Supports W&B logging when enabled in config.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import torch
import numpy as np

from graph_metanet.configs.config import (
    ExperimentConfig,
    load_config,
    save_config,
    config_to_dict,
)
from graph_metanet.models.learned_optimizer import GraphMetaOptimizer
from graph_metanet.tasks.task_library import build_task
from graph_metanet.meta_training.trainer import meta_train_step, evaluate_optimizer

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer_from_config(config: ExperimentConfig) -> GraphMetaOptimizer:
    """Instantiate a GraphMetaOptimizer from config."""
    mc = config.model
    return GraphMetaOptimizer(
        node_hidden_dim=mc.node_hidden_dim,
        edge_hidden_dim=mc.edge_hidden_dim,
        gnn_hidden_dim=mc.gnn_hidden_dim,
        num_gnn_layers=mc.num_gnn_layers,
        num_mp_rounds=mc.num_mp_rounds,
        use_layer_norm=mc.use_layer_norm,
        use_per_param_heads=mc.use_per_param_heads,
        output_scale=mc.output_scale,
        use_hidden_state=mc.use_hidden_state,
        hidden_state_dim=mc.hidden_state_dim,
    )


def build_meta_optimizer(
    optimizer: GraphMetaOptimizer, config: ExperimentConfig
) -> torch.optim.Optimizer:
    """Build the meta-optimizer for the learned optimizer's parameters."""
    tc = config.meta_training
    if tc.meta_optimizer == "adam":
        return torch.optim.Adam(
            optimizer.parameters(), lr=tc.meta_lr, weight_decay=tc.meta_weight_decay
        )
    elif tc.meta_optimizer == "sgd":
        return torch.optim.SGD(
            optimizer.parameters(), lr=tc.meta_lr, weight_decay=tc.meta_weight_decay
        )
    else:
        raise ValueError(f"Unknown meta-optimizer: {tc.meta_optimizer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Graph MetaNetwork Learned Optimizer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--overrides", type=str, default=None,
                        help="JSON string of config overrides")
    args = parser.parse_args()

    # Load config
    overrides = None
    if args.overrides:
        import json
        overrides = json.loads(args.overrides)

    config = load_config(args.config, overrides)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Setup output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(output_dir / "config.yaml"))

    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Output directory: {output_dir}")

    # Set seed
    set_seed(config.seed)

    # Device
    device = torch.device(config.device)

    # Initialize W&B
    wandb_run = None
    if config.wandb.enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.run_name or config.experiment_name,
                config=config_to_dict(config),
                tags=config.wandb.tags,
                notes=config.wandb.notes,
                dir=str(output_dir),
            )
            logger.info(f"W&B run: {wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed; disabling W&B logging")
            wandb_run = None
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            wandb_run = None

    # Build learned optimizer
    optimizer = build_optimizer_from_config(config).to(device)
    num_params = sum(p.numel() for p in optimizer.parameters())
    logger.info(f"Learned optimizer parameters: {num_params:,}")

    # Build meta-optimizer
    meta_opt = build_meta_optimizer(optimizer, config)

    # Build tasks
    task_specs = []
    for task_name in config.tasks.task_names:
        task_params = config.tasks.task_params.get(task_name, {})
        task_specs.append(build_task(task_name, **task_params))
        logger.info(f"Task: {task_specs[-1].name}")

    # Meta-training loop
    tc = config.meta_training
    for outer_step in range(1, tc.num_outer_steps + 1):
        # Sample a random task
        task_idx = random.randint(0, len(task_specs) - 1)
        task_spec = task_specs[task_idx]

        metrics = meta_train_step(
            optimizer=optimizer,
            meta_optimizer=meta_opt,
            task_spec=task_spec,
            num_inner_steps=tc.num_inner_steps,
            truncation_length=tc.truncation_length,
            device=device,
        )

        if outer_step % 10 == 0:
            logger.info(
                f"Step {outer_step}/{tc.num_outer_steps} "
                f"[{task_spec.name}] "
                f"inner_loss: {metrics['inner_loss_start']:.4f} -> {metrics['inner_loss_end']:.4f} "
                f"meta_loss: {metrics['meta_loss']:.4f}"
            )

        if wandb_run is not None:
            import wandb
            log_dict = {
                f"train/{k}": v for k, v in metrics.items()
            }
            log_dict["train/task"] = task_spec.name
            log_dict["train/outer_step"] = outer_step
            wandb.log(log_dict, step=outer_step)

        # Evaluation
        if outer_step % tc.eval_every == 0:
            logger.info("Running evaluation...")
            for ts in task_specs:
                eval_losses = []
                for _ in range(tc.num_eval_tasks):
                    losses = evaluate_optimizer(
                        optimizer, ts, tc.num_eval_steps, device
                    )
                    eval_losses.append(losses[-1] if losses else 0.0)

                avg_final_loss = sum(eval_losses) / len(eval_losses)
                logger.info(f"  Eval [{ts.name}]: avg final loss = {avg_final_loss:.4f}")

                if wandb_run is not None:
                    wandb.log({
                        f"eval/{ts.name}/avg_final_loss": avg_final_loss,
                    }, step=outer_step)

        # Save checkpoint
        if outer_step % tc.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_{outer_step}.pt"
            torch.save({
                "outer_step": outer_step,
                "optimizer_state_dict": optimizer.state_dict(),
                "meta_optimizer_state_dict": meta_opt.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = output_dir / "final_model.pt"
    torch.save(optimizer.state_dict(), final_path)
    logger.info(f"Saved final model: {final_path}")

    if wandb_run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
