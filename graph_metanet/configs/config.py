"""Configuration loading and management.

Provides YAML-based configuration for all experiments with support for
hierarchical configs, defaults, and overrides.
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for the GraphMetaOptimizer architecture."""
    node_hidden_dim: int = 64
    edge_hidden_dim: int = 64
    gnn_hidden_dim: int = 128
    num_gnn_layers: int = 3
    num_mp_rounds: int = 2
    use_layer_norm: bool = True
    use_per_param_heads: bool = False
    output_scale: float = 0.001
    use_hidden_state: bool = True
    hidden_state_dim: int = 32


@dataclass
class MetaTrainingConfig:
    """Configuration for the meta-training procedure."""
    meta_lr: float = 1e-3
    meta_optimizer: str = "adam"
    meta_weight_decay: float = 0.0
    num_outer_steps: int = 1000
    num_inner_steps: int = 20
    truncation_length: int = 10
    grad_clip_norm: float = 1.0
    eval_every: int = 50
    save_every: int = 200
    num_eval_tasks: int = 5
    num_eval_steps: int = 50


@dataclass
class TaskConfig:
    """Configuration for meta-training tasks."""
    task_names: List[str] = field(default_factory=lambda: ["quadratic", "mlp_regression"])
    task_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""
    enabled: bool = True
    project: str = "graph-metanet-optimizer"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    experiment_name: str = "default"
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    meta_training: MetaTrainingConfig = field(default_factory=MetaTrainingConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _merge_dicts(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, overrides: Optional[Dict] = None) -> ExperimentConfig:
    """Load experiment configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.
        overrides: Optional dict of overrides to apply on top of the YAML.

    Returns:
        An ExperimentConfig instance.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _merge_dicts(raw, overrides)

    model_cfg = ModelConfig(**raw.get("model", {}))
    meta_cfg = MetaTrainingConfig(**raw.get("meta_training", {}))
    task_cfg_raw = raw.get("tasks", {})
    default_task = TaskConfig()
    task_cfg = TaskConfig(
        task_names=task_cfg_raw.get("task_names", default_task.task_names),
        task_params=task_cfg_raw.get("task_params", {}),
    )
    wandb_cfg = WandbConfig(**raw.get("wandb", {}))

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "default"),
        seed=raw.get("seed", 42),
        device=raw.get("device", "cpu"),
        output_dir=raw.get("output_dir", "outputs"),
        model=model_cfg,
        meta_training=meta_cfg,
        tasks=task_cfg,
        wandb=wandb_cfg,
    )


def save_config(config: ExperimentConfig, path: str) -> None:
    """Save an ExperimentConfig to a YAML file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)


def config_to_dict(config: ExperimentConfig) -> Dict:
    """Convert config to a flat dict suitable for W&B logging."""
    return asdict(config)
